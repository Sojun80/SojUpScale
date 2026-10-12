#!/usr/bin/env python3
"""Upscale old SD video with SeedVR2.

The source range is split into segments (about --segment-seconds long, cut at
scene changes where one is close so the joins don't show). Each segment:
  1. prep     exact frame range, square pixels, lossless H.264
  2. seedvr2  the upstream CLI, PNG frames out (8-bit RGB)
  3. encode   RGB -> BT.709 10-bit x265 (on the CPU, overlapping the next
              segment's SeedVR2 run); its PNGs are deleted afterwards
Then the segments are joined and the source audio muxed back, and for short
ranges a Lanczos-vs-SeedVR2 side-by-side is made.

Rerunning with the same --work resumes: finished segments are skipped.

Interlaced or telecined sources are refused for now: deinterlacing/IVTC is
not implemented yet, and feeding combed frames to the model bakes combing in.

Stdlib only; runs on thed1 with the venv and models from thed1/setup.sh.
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from fractions import Fraction
from pathlib import Path

DEPS = Path(os.environ.get("VIDEOUPSCALE_DEPS", Path.home() / "VideoUpscale-deps"))
SCENE_THRESHOLD = 10      # ffmpeg scdet score (0-100) that counts as a cut
SCENE_WINDOW = 0.25       # look for a cut within this fraction of a segment of the target
COMPARE_MAX_SECONDS = 120


def show(cmd):
    print("+ " + " ".join(str(c) for c in cmd), flush=True)
    return [str(c) for c in cmd]


def run(cmd, log=None):
    cmd = show(cmd)
    if log is None:
        subprocess.run(cmd, check=True)
        return
    with open(log, "w") as f:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                bufsize=1, env={**os.environ, "PYTHONUNBUFFERED": "1"})
        for line in proc.stdout:
            sys.stdout.write(line)
            f.write(line)
        if proc.wait() != 0:
            sys.exit(f"command failed ({proc.returncode}), see {log}")


def probe(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets", "-show_entries",
         "stream=width,height,sample_aspect_ratio,r_frame_rate,nb_read_packets",
         "-of", "json", str(path)], check=True, capture_output=True, text=True).stdout
    s = json.loads(out)["streams"][0]
    sar = s.get("sample_aspect_ratio", "1:1")
    sar = Fraction(1) if sar in ("0:1", "N/A") else Fraction(*map(int, sar.split(":")))
    has_audio = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries",
         "stream=index", "-of", "csv=p=0", str(path)],
        check=True, capture_output=True, text=True).stdout.strip() != ""
    return dict(w=s["width"], h=s["height"], sar=sar, fps=Fraction(s["r_frame_rate"]),
                frames=int(s["nb_read_packets"]), audio=has_audio)


def frame_count(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets",
                          "-show_entries", "stream=nb_read_packets", "-of", "csv=p=0", str(path)],
                         capture_output=True, text=True).stdout.strip()
    return int(out) if out.isdigit() else -1


def interlace_check(src, start):
    """Fraction of frames idet calls interlaced, over 30 s from `start`."""
    r = subprocess.run(
        ["ffmpeg", "-hide_banner", "-nostats", "-ss", str(start), "-i", str(src), "-t", "30",
         "-an", "-vf", "idet", "-f", "null", "-"], capture_output=True, text=True)
    for line in r.stderr.splitlines():
        if "Multi frame detection" in line:
            nums = [int(t) for t in line.split() if t.isdigit()]
            tff, bff, prog = nums[0], nums[1], nums[2]
            return (tff + bff) / max(1, tff + bff + prog)
    return 0.0


def scene_cuts(src, fps, first, last):
    """Frame numbers (absolute) where a new shot starts, within [first, last)."""
    r = subprocess.run(
        ["ffmpeg", "-hide_banner", "-nostats", "-ss", f"{seek_time(first, fps):.6f}", "-i", str(src),
         "-frames:v", str(last - first), "-an", "-vf", f"scdet=threshold={SCENE_THRESHOLD}",
         "-f", "null", "-"], capture_output=True, text=True)
    times = [float(t) for t in re.findall(r"lavfi\.scd\.time:\s*([\d.]+)", r.stderr)]
    return sorted({first + round(t * fps) for t in times})


def plan_segments(first, last, seg_frames, cuts):
    """Split [first, last) into pieces near seg_frames long, snapping ends to scene cuts."""
    bounds = [first]
    window = int(seg_frames * SCENE_WINDOW)
    while last - bounds[-1] > seg_frames + window:
        target = bounds[-1] + seg_frames
        near = [c for c in cuts if abs(c - target) <= window and c - bounds[-1] > window]
        bounds.append(min(near, key=lambda c: abs(c - target)) if near else target)
    bounds.append(last)
    return list(zip(bounds[:-1], bounds[1:]))


def seek_time(frame, fps):
    # Half a frame early, so the seek lands exactly on `frame` despite timestamp rounding.
    return 0.0 if frame == 0 else float((frame - Fraction(1, 2)) / fps)


def even(x):
    return int(round(x / 2)) * 2


def square_size(info):
    """Square-pixel size, never throwing away source pixels."""
    w, h, sar = info["w"], info["h"], info["sar"]
    if sar >= 1:
        return even(w * sar), h
    return w, even(h / sar)


def prep_cmd(src, fps, sq, start, n, out):
    # Untagged SD: decoded as BT.601 (ffmpeg's default for SD), 4:2:0 kept, lossless so the
    # model sees exactly the source pixels, rescaled to square.
    return ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-ss", f"{seek_time(start, fps):.6f}",
            "-i", src, "-frames:v", n, "-map", "0:v:0", "-an",
            "-vf", f"scale={sq[0]}:{sq[1]}:flags=lanczos,setsar=1",
            "-c:v", "libx264", "-qp", "0", "-preset", "veryfast", "-pix_fmt", "yuv420p",
            "-color_primaries", "smpte170m", "-color_trc", "smpte170m", "-colorspace", "smpte170m", out]


def encode_cmd(frame_dir, fps, crf, out):
    return ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-framerate", fps,
            "-pattern_type", "glob", "-i", f"{frame_dir}/*.png",
            "-vf", "scale=out_color_matrix=bt709:out_range=tv,format=yuv420p10le",
            "-c:v", "libx265", "-crf", crf, "-preset", "slow",
            "-x265-params", "colorprim=bt709:transfer=bt709:colormatrix=bt709:log-level=warning",
            "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709",
            "-color_range", "tv", out]


def frame_dir_of(frames):
    pngs = sorted(frames.rglob("*.png"))
    dirs = {p.parent for p in pngs}
    if len(dirs) != 1:
        sys.exit(f"expected PNG frames in one directory under {frames}, found {len(pngs)} in {dirs}")
    return dirs.pop(), len(pngs)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src")
    ap.add_argument("--work", required=True, help="work/output directory (reuse it to resume)")
    ap.add_argument("--start", type=float, default=0, help="range start, seconds")
    ap.add_argument("--duration", type=float, default=0, help="range length, seconds (0 = to end)")
    ap.add_argument("--segment-seconds", type=float, default=60)
    ap.add_argument("--height", type=int, default=1080, help="output height (short side)")
    ap.add_argument("--model", default="seedvr2_ema_7b_fp16.safetensors")
    ap.add_argument("--gpus", default="0,1,2,3", help="CUDA device list, e.g. 2 or 0,1,2,3")
    ap.add_argument("--batch-size", type=int, default=45, help="frames per batch, 4n+1")
    ap.add_argument("--overlap", type=int, default=3, help="temporal overlap between batches/GPUs")
    ap.add_argument("--color-correction", default="lab")
    ap.add_argument("--attention", default="sdpa")
    ap.add_argument("--crf", type=int, default=16)
    ap.add_argument("--compare", choices=["auto", "yes", "no"], default="auto",
                    help=f"side-by-side video (auto: ranges up to {COMPARE_MAX_SECONDS} s)")
    ap.add_argument("--keep-frames", action="store_true", help="keep each segment's PNGs")
    ap.add_argument("--allow-interlaced", action="store_true")
    a = ap.parse_args()

    if (a.batch_size - 1) % 4:
        sys.exit("--batch-size must be 4n+1 (1, 5, 9, 13, 17, 21, ...)")
    src = Path(a.src).resolve()
    work = Path(a.work).expanduser().resolve()
    work.mkdir(parents=True, exist_ok=True)
    info = probe(src)
    fps = info["fps"]
    sq = square_size(info)
    out_h = a.height
    out_w = even(sq[0] * out_h / sq[1])
    first = round(a.start * fps)
    last = min(info["frames"], first + round(a.duration * fps)) if a.duration else info["frames"]
    print(f"source {info['w']}x{info['h']} SAR {info['sar']} @ {fps} ({info['frames']} frames) -> "
          f"square {sq[0]}x{sq[1]} -> output {out_w}x{out_h}; range frames {first}-{last}", flush=True)

    plan_file = work / "segments.json"
    if plan_file.exists():
        plan = json.loads(plan_file.read_text())
        if (plan["first"], plan["last"]) != (first, last):
            sys.exit(f"{work} holds a different range ({plan['first']}-{plan['last']}); use a new --work")
        segments = [tuple(s) for s in plan["segments"]]
    else:
        ratio = interlace_check(src, a.start)
        print(f"idet: {ratio:.1%} of frames interlaced", flush=True)
        if ratio > 0.05 and not a.allow_interlaced:
            sys.exit("source looks interlaced/telecined; deinterlace first (not implemented yet) "
                     "or pass --allow-interlaced")
        cuts = scene_cuts(src, fps, first, last)
        segments = plan_segments(first, last, round(a.segment_seconds * fps), cuts)
        plan_file.write_text(json.dumps(dict(first=first, last=last, scene_cuts=cuts,
                                             segments=segments), indent=1))
    print(f"{len(segments)} segments: " + ", ".join(f"{s}-{e}" for s, e in segments), flush=True)

    pending = None  # (encode process, segment dir, frames dir, expected frames)

    def finish(p):
        proc, seg, frames, n = p
        if proc.wait() != 0:
            sys.exit(f"x265 encode failed for {seg}")
        got = frame_count(seg / "video.mkv")
        if got != n:
            sys.exit(f"{seg}/video.mkv has {got} frames, expected {n}")
        (seg / "done").write_text("ok\n")
        if not a.keep_frames:
            shutil.rmtree(frames, ignore_errors=True)
            (seg / "prep.mp4").unlink(missing_ok=True)
        print(f"segment {seg.name} encoded ({n} frames)", flush=True)

    t_all = time.time()
    gpu_secs = 0.0
    for i, (s, e) in enumerate(segments):
        seg = work / f"seg{i:03d}"
        if (seg / "done").exists():
            continue
        seg.mkdir(exist_ok=True)
        frames = seg / "frames"
        run(prep_cmd(src, fps, sq, s, e - s, seg / "prep.mp4"))
        if frame_count(seg / "prep.mp4") != e - s:
            sys.exit(f"{seg}/prep.mp4 does not have {e - s} frames")
        if frames.exists():
            shutil.rmtree(frames)
        t0 = time.time()
        run([DEPS / "venv/bin/python", DEPS / "seedvr2/inference_cli.py", seg / "prep.mp4",
             "--output", frames, "--output_format", "png",
             "--model_dir", DEPS / "models/SEEDVR2", "--dit_model", a.model,
             "--resolution", out_h, "--batch_size", a.batch_size,
             "--temporal_overlap", a.overlap, "--color_correction", a.color_correction,
             "--attention_mode", a.attention, "--cuda_device", a.gpus], log=seg / "seedvr2.log")
        secs = time.time() - t0
        gpu_secs += secs
        frame_dir, n = frame_dir_of(frames)
        if n != e - s:
            sys.exit(f"SeedVR2 wrote {n} frames for {seg}, expected {e - s}")
        print(f"segment {seg.name}: {n} frames in {secs:.0f} s = {n / secs:.3f} fps", flush=True)
        (seg / "timing.json").write_text(json.dumps(dict(frames=n, seconds=round(secs, 1)), indent=1))
        if pending:
            finish(pending)
        pending = (subprocess.Popen(show(encode_cmd(frame_dir, fps, a.crf, seg / "video.mkv"))),
                   seg, frames, n)
    if pending:
        finish(pending)

    # Join segments, then add the source audio for the same range.
    listing = work / "segments.txt"
    listing.write_text("".join(f"file '{work / f'seg{i:03d}' / 'video.mkv'}'\n" for i in range(len(segments))))
    video = work / "video.mkv"
    run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "concat", "-safe", "0",
         "-i", listing, "-c", "copy", video])
    result = work / "upscaled.mkv"
    whole = first == 0 and last == info["frames"]
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", video]
    if info["audio"]:
        cut = [] if whole else ["-ss", f"{float(first / fps):.6f}", "-t", f"{float((last - first) / fps):.6f}"]
        # A partial range re-encodes the audio (a stream copy can't cut on our frame boundary).
        acodec = ["-c:a", "copy"] if whole else ["-c:a", "aac", "-b:a", "256k"]
        cmd += [*cut, "-i", src, "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", *acodec, "-shortest"]
    else:
        cmd += ["-c", "copy"]
    run([*cmd, result])
    total = frame_count(result)
    print(f"upscaled.mkv: {total} frames (expected {last - first})", flush=True)
    timing = dict(frames=last - first, segments=len(segments), wall_seconds=round(time.time() - t_all, 1),
                  seedvr2_seconds_this_run=round(gpu_secs, 1), gpus=a.gpus, model=a.model,
                  batch_size=a.batch_size, overlap=a.overlap, size=f"{out_w}x{out_h}")
    (work / "timing.json").write_text(json.dumps(timing, indent=2))

    if a.compare == "yes" or (a.compare == "auto" and (last - first) / fps <= COMPARE_MAX_SECONDS):
        # Left: plain Lanczos of the source (square pixels, BT.709). Right: SeedVR2.
        run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
             "-ss", f"{seek_time(first, fps):.6f}", "-i", src, "-i", result, "-filter_complex",
             f"[0:v]trim=end_frame={last - first},scale={out_w}:{out_h}:flags=lanczos:"
             f"in_color_matrix=bt601:out_color_matrix=bt709,format=yuv420p,setsar=1[l];"
             f"[1:v]scale=in_color_matrix=bt709:out_color_matrix=bt709,format=yuv420p,setsar=1[r];"
             f"[l][r]hstack=inputs=2[v]", "-map", "[v]",
             "-c:v", "libx264", "-crf", "14", "-preset", "medium", "-colorspace", "bt709",
             "-color_primaries", "bt709", "-color_trc", "bt709", work / "compare.mp4"])
    print(f"done: {work}", flush=True)


if __name__ == "__main__":
    main()
