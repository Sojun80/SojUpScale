"""engines/seedvr2_runner.py + engines/framework.py against a fake upstream
(tests/fake_deps): per-piece batch sizes, prepend trimming, crossfades between
pieces of a long shot, 16-bit output, and the orchestrator around them."""
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

from tests.util import HAVE_FFMPEG, REPO, make_clip, upscale

try:
    import cv2
    import numpy as np
except ImportError:
    cv2 = None

sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "engines"))
from sojupscale import media, plan  # noqa: E402

MODEL = "seedvr2_ema_7b_fp16.safetensors"


@unittest.skipUnless(HAVE_FFMPEG and cv2 is not None, "needs ffmpeg, numpy and opencv")
class FakeSeedVR2(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp())
        cls.deps = cls.tmp / "deps"
        shutil.copytree(REPO / "tests/fake_deps/seedvr2", cls.deps / "seedvr2")
        (cls.deps / "venv/bin").mkdir(parents=True)
        (cls.deps / "venv/bin/python").symlink_to(sys.executable)
        (cls.deps / "models/SEEDVR2").mkdir(parents=True)
        (cls.deps / "models/SEEDVR2" / MODEL).write_bytes(b"")
        cls.src = cls.tmp / "sd.mp4"
        make_clip(cls.src, audio=False)
        cls.log = cls.tmp / "calls.jsonl"
        os.environ["VIDEOUPSCALE_DEPS"] = str(cls.deps)
        os.environ["FAKE_SEEDVR2_LOG"] = str(cls.log)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_pieces_batches_and_crossfades(self):
        work = self.tmp / "w"
        r = upscale(self.src, "--work", work, "--engine", "seedvr2", "--gpus", "0,1", "--height", "540",
                    "--segment-seconds", "6", "--max-piece-seconds", "2", "--piece-overlap", "12",
                    "--batch-size", "21", "--prepend-frames", "2", "--x265-preset", "ultrafast",
                    "--keep-frames", "--compare", "no")
        self.assertEqual(r.returncode, 0, r.stdout[-4000:] + r.stderr[-4000:])
        p = json.loads((work / "plan.json").read_text())
        calls = [json.loads(line) for line in self.log.read_text().splitlines()]
        pieces = [pc for job in p["jobs"] for pc in job]
        self.assertEqual(len(calls), len(pieces))
        for c in calls:  # prepend frames reach upstream; batch size balanced for that length
            self.assertEqual(c["prepend"], 2)
            self.assertEqual(c["batch_size"], plan.balanced_batch(c["frames"], 21, 3))
        info = media.probe(self.src)
        self.assertEqual(media.probe(work / "upscaled.mkv")["frames"], info["frames"])

        # The mandelbrot shot (long, moving) is split into crossfaded pieces in one job.
        job_no, job = next((k, j) for k, j in enumerate(p["jobs"]) if len(j) > 1 and j[0]["blend_out"])
        frames = work / "jobs" / f"j{job_no:03d}" / "frames"
        self.assertFalse(any((work / "jobs" / f"j{job_no:03d}" / "tail").glob("*.png")))
        png = cv2.imread(str(frames / f"f{job[0]['keep_start']:07d}.png"), cv2.IMREAD_UNCHANGED)
        self.assertEqual(png.dtype, np.uint16)

        # Which offset each piece got: the fake alternates per call within a worker.
        by_pid = {}
        for c in calls:
            by_pid.setdefault(c["pid"], []).append(c)
        # The job's pieces were the consecutive calls of one worker; find them by frame counts.
        n_frames = [pc["end"] - pc["start"] + 2 for pc in job]
        offsets = None
        for seq in by_pid.values():
            for s in range(len(seq) - len(job) + 1):
                if [c["frames"] for c in seq[s:s + len(job)]] == n_frames:
                    offsets = [c["offset"] for c in seq[s:s + len(job)]]
        self.assertIsNotNone(offsets)
        import framework
        spec = dict(source=dict(path=str(self.src), square=[720, 540], deinterlace="none",
                                info=dict(fps=str(info["fps"]), matrix=info["matrix"], range=info["range"])))
        for k, pc in enumerate(job):
            ref = framework.read_frames(spec, pc)
            w_prev = expected_weights(pc["blend_in"])
            for n in range(pc["keep_start"], pc["keep_end"] - pc["blend_out"]):
                cubic = cv2.resize(ref[n - pc["start"]], (720, 540), interpolation=cv2.INTER_CUBIC)
                got = framework.read_png(frames / f"f{n:07d}.png")
                j = n - pc["keep_start"]
                want = offsets[k] if j >= pc["blend_in"] else \
                    offsets[k - 1] * w_prev[j] + offsets[k] * (1 - w_prev[j])
                d = float(np.median(got - np.clip(cubic, 0, 1)))
                self.assertAlmostEqual(d, want, delta=0.004, msg=f"frame {n} piece {k}")


def expected_weights(k):
    """SeedVR2's overlap curve, written out independently of framework.blend_weights."""
    import math
    out = []
    for j in range(k):
        u = min(1.0, max(0.0, (j / (k - 1) - 1 / 3) * 3))
        out.append(0.5 + 0.5 * math.cos(math.pi * u))
    return out


class BlendWeights(unittest.TestCase):
    @unittest.skipIf(cv2 is None, "needs numpy and opencv")
    def test_curve(self):
        import framework
        w = framework.blend_weights(12)
        for a, b in zip(w, expected_weights(12)):
            self.assertAlmostEqual(float(a), b, places=6)
        self.assertTrue(all(a >= b for a, b in zip(w, w[1:])))
        self.assertEqual((w[:4] == 1).sum(), 4)  # outer thirds use one side only
        self.assertEqual((w[-4:] == 0).sum(), 4)
        self.assertEqual(list(framework.blend_weights(2)), [1.0, 0.0])

    @unittest.skipIf(cv2 is None, "needs numpy and opencv")
    def test_png_rounding(self):
        import framework
        x = np.full((2, 2, 3), 0.5, np.float32)
        self.assertEqual(int(framework.to_png_array(x, 8)[0, 0, 0]), 128)  # truncation gave 127
        self.assertEqual(int(framework.to_png_array(x, 16)[0, 0, 0]), 32768)


if __name__ == "__main__":
    unittest.main()
