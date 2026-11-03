"""End-to-end runs of upscale.py with the CPU lanczos engine (needs ffmpeg)."""
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from tests.util import HAVE_FFMPEG, REPO, make_clip, per_frame_psnr, upscale

import sys
sys.path.insert(0, str(REPO))
from sojupscale import media  # noqa: E402

FAST = ["--engine", "lanczos", "--height", "540", "--segment-seconds", "2", "--max-piece-seconds", "1",
        "--piece-overlap", "6", "--workers", "2", "--encoders", "2", "--x265-preset", "ultrafast",
        "--crf", "4"]


@unittest.skipUnless(HAVE_FFMPEG, "ffmpeg not installed")
class Pipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp())
        cls.src = cls.tmp / "sd.mp4"
        cls.cuts = make_clip(cls.src)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def run_ok(self, *args):
        r = upscale(*args)
        self.assertEqual(r.returncode, 0, r.stdout[-3000:] + r.stderr[-3000:])
        return r

    def test_whole_file_frames_aligned_resume_and_refusal(self):
        work = self.tmp / "w_whole"
        self.run_ok(self.src, "--work", work, *FAST)
        info = media.probe(self.src)
        out = media.probe(work / "upscaled.mkv")
        self.assertEqual(out["frames"], info["frames"])
        self.assertEqual((out["w"], out["h"]), (720, 540))
        self.assertTrue(out["audio"])
        p = json.loads((work / "plan.json").read_text())
        self.assertEqual(p["scene_cuts"], self.cuts)
        self.assertGreater(sum(len(j) for j in p["jobs"]), len(p["jobs"]))  # long shot was split
        # Every output frame matches the same source frame (no shifts at piece/job joins).
        ref = media.to_rgb_filter(info, (720, 540))
        psnr = per_frame_psnr(work / "upscaled.mkv", self.src, filter_b=ref + ",scale=out_color_matrix=bt709")
        self.assertEqual(len(psnr), info["frames"])
        self.assertGreater(min(psnr), 38, psnr)
        # Resume: nothing left to do.
        r = self.run_ok(self.src, "--work", work, *FAST)
        self.assertNotIn("encoded (", r.stdout)
        # A setting that changes pixels is refused instead of silently reused.
        r = upscale(self.src, "--work", work, *FAST, "--height", "720")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("output", r.stderr)
        # --restart discards and redoes.
        self.run_ok(self.src, "--work", work, *FAST, "--height", "720", "--restart", "--compare", "no")
        self.assertEqual(media.probe(work / "upscaled.mkv")["h"], 720)

    def test_partial_range_starts_on_the_right_frame(self):
        work = self.tmp / "w_part"
        info = media.probe(self.src)
        first = 61  # one frame after the first cut
        self.run_ok(self.src, "--work", work, *FAST, "--start", f"{first / float(info['fps']):.6f}",
                    "--duration", "3")
        p = json.loads((work / "plan.json").read_text())
        self.assertEqual(p["first"], first)
        out = media.probe(work / "upscaled.mkv")
        self.assertEqual(out["frames"], p["last"] - first)
        ref = media.to_rgb_filter(info, (720, 540))
        psnr = per_frame_psnr(work / "upscaled.mkv", self.src,
                              filter_b=f"trim=start_frame={first},setpts=PTS-STARTPTS,{ref},"
                                       f"scale=out_color_matrix=bt709")
        self.assertGreater(min(psnr[:out["frames"]]), 38, psnr)

    def test_replaced_source_is_refused(self):
        work = self.tmp / "w_src"
        src2 = self.tmp / "other.mp4"
        self.run_ok(self.src, "--work", work, *FAST, "--compare", "no")
        make_clip(src2, seconds=(1, 5, 2.5))
        shutil.copy(src2, self.tmp / "sd_copy.mp4")
        r = upscale(self.tmp / "sd_copy.mp4", "--work", work, *FAST, "--compare", "no")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("source", r.stderr)

    def test_interrupted_job_resumes_from_its_pieces(self):
        work = self.tmp / "w_int"
        self.run_ok(self.src, "--work", work, *FAST, "--compare", "no", "--keep-frames")
        # Simulate a crash: job 1 lost its last piece and was never encoded.
        j = work / "jobs" / "j001"
        markers = sorted((j / "pieces").glob("p*.json"))
        markers[-1].unlink()
        for f in ("done", "encoded", "video.mkv"):
            (j / f).unlink()
        r = self.run_ok(self.src, "--work", work, *FAST, "--compare", "no")
        self.assertIn("job j001 encoded", r.stdout)
        self.assertIn(f"piece {len(markers) - 1}:", r.stdout)  # only the missing piece reran
        self.assertNotIn("piece 0:", r.stdout)


if __name__ == "__main__":
    unittest.main()
