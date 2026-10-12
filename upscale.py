#!/usr/bin/env python3
"""Upscale old SD video with SeedVR2.

Pipeline (each step writes into the work directory):
  1. prep     cut the excerpt, convert to square pixels, lossless H.264
  2. seedvr2  run the upstream CLI, PNG frames out (8-bit RGB)
  3. encode   RGB -> BT.709 10-bit x265, tagged, original audio muxed back
  4. compare  Lanczos baseline next to the SeedVR2 result, for judging

Interlaced or telecined sources are refused for now: deinterlacing/IVTC is
not implemented yet, and feeding combed frames to the model bakes combing in.

Stdlib only; runs on thed1 with the venv and models from thed1/setup.sh.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from fractions import Fraction
from pathlib import Path

DEPS = Path(os.environ.get("VIDEOUPSCALE_DEPS", Path.home() / "VideoUpscale-deps"))


def run(cmd, log=None):
    print("+ " + " ".join(str(c) for c in cmd), flush=True)
    if log is None:
        subprocess.run([str(c) for c in cmd], check=True)
        return
    with open(log, "w") as f:
        proc = subprocess.Popen([str(c) for c in cmd], stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, bufsize=1,
                                env={**os.environ, "PYTHONUNBUFFERED": "1"})
        for line in proc.stdout:
            sys.stdout.write(line)
            f.write(line)
        if proc.wait() != 0:
            sys.exit(f"command failed ({proc.returncode}), see {log}")


def probe(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height,sample_aspect_ratio,r_frame_rate,field_order",
         "-of", "json", str(path)], check=True, capture_output=True, text=True).stdout
    s = json.loads(out)["streams"][0]
    sar = s.get("sample_aspect_ratio", "1:1")
    sar = Fraction(1) if sar in ("0:1", "N/A") else Fraction(*map(int, sar.split(":")))
    has_audio = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries",
         "stream=index", "-of", "csv=p=0", str(path)],
        check=True, capture_output=True, text=True).stdout.strip() != ""
    return dict(w=s["width"], h=s["height"], sar=sar, fps=s["r_frame_rate"],
                field_order=s.get("field_order", "unknown"), audio=has_audio)


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


def even(x):
    return int(round(x / 2)) * 2


def square_size(info):
    """Square-pixel size, never throwing away source pixels."""
    w, h, sar = info["w"], info["h"], info["sar"]
    if sar >= 1:
        return even(w * sar), h
    return w, even(h / sar)


def cut_args(start):
    return ["-ss", str(start)] if start else []


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src")
    ap.add_argument("--work", required=True, help="work/output directory")
    ap.add_argument("--start", type=float, default=0, help="excerpt start, seconds")
    ap.add_argument("--duration", type=float, default=0, help="excerpt length, seconds (0 = to end)")
    ap.add_argument("--height", type=int, default=1080, help="output height (short side)")
    ap.add_argument("--model", default="seedvr2_ema_7b_fp16.safetensors")
    ap.add_argument("--gpus", default="2", help="CUDA device list, e.g. 2 or 0,1,2,3")
    ap.add_argument("--batch-size", type=int, default=21, help="frames per batch, 4n+1")
    ap.add_argument("--overlap", type=int, default=3, help="temporal overlap between batches/GPUs")
    ap.add_argument("--color-correction", default="lab")
    ap.add_argument("--attention", default="sdpa")
    ap.add_argument("--crf", type=int, default=16)
    ap.add_argument("--steps", default="prep,seedvr2,encode,compare")
    ap.add_argument("--allow-interlaced", action="store_true")
    a = ap.parse_args()

    if (a.batch_size - 1) % 4:
        sys.exit("--batch-size must be 4n+1 (1, 5, 9, 13, 17, 21, ...)")
    src = Path(a.src).resolve()
    work = Path(a.work).expanduser().resolve()
    work.mkdir(parents=True, exist_ok=True)
    steps = set(a.steps.split(","))
    info = probe(src)
    sq_w, sq_h = square_size(info)
    out_h = a.height
    out_w = even(sq_w * out_h / sq_h)
    cut = cut_args(a.start)
    dur = ["-t", str(a.duration)] if a.duration else []
    prep = work / "prep.mp4"
    frames = work / "frames"
    result = work / "upscaled.mkv"
    print(f"source {info['w']}x{info['h']} SAR {info['sar']} @ {info['fps']} -> "
          f"square {sq_w}x{sq_h} -> output {out_w}x{out_h}", flush=True)

    if "prep" in steps:
        ratio = interlace_check(src, a.start)
        print(f"idet: {ratio:.1%} of frames interlaced", flush=True)
        if ratio > 0.05 and not a.allow_interlaced:
            sys.exit("source looks interlaced/telecined; deinterlace first (not implemented yet) "
                     "or pass --allow-interlaced")
        # Source is untagged SD: decode as BT.601 (ffmpeg's default for SD), keep 4:2:0,
        # lossless so the model sees exactly the source pixels, rescaled to square.
        run(["ffmpeg", "-hide_banner", "-y", *cut, "-i", src, *dur, "-map", "0:v:0", "-an",
             "-vf", f"scale={sq_w}:{sq_h}:flags=lanczos,setsar=1",
             "-c:v", "libx264", "-qp", "0", "-preset", "veryfast", "-pix_fmt", "yuv420p",
             "-color_primaries", "smpte170m", "-color_trc", "smpte170m", "-colorspace", "smpte170m",
             prep])

    if "seedvr2" in steps:
        if frames.exists():
            shutil.rmtree(frames)
        cli = [DEPS / "venv/bin/python", DEPS / "seedvr2/inference_cli.py", prep,
               "--output", frames, "--output_format", "png",
               "--model_dir", DEPS / "models/SEEDVR2", "--dit_model", a.model,
               "--resolution", out_h, "--batch_size", a.batch_size,
               "--temporal_overlap", a.overlap, "--color_correction", a.color_correction,
               "--attention_mode", a.attention, "--cuda_device", a.gpus]
        t0 = time.time()
        run(cli, log=work / "seedvr2.log")
        n = len(list(frames.rglob("*.png")))
        secs = time.time() - t0
        print(f"seedvr2: {n} frames in {secs:.0f} s = {n / secs:.3f} fps", flush=True)
        (work / "timing.json").write_text(json.dumps(
            dict(frames=n, seconds=round(secs, 1), fps=round(n / secs, 4), gpus=a.gpus,
                 model=a.model, batch_size=a.batch_size, overlap=a.overlap,
                 size=f"{out_w}x{out_h}"), indent=2))

    if "encode" in steps:
        pngs = sorted(frames.rglob("*.png"))
        if not pngs:
            sys.exit(f"no PNG frames under {frames}")
        dirs = {p.parent for p in pngs}
        if len(dirs) != 1:
            sys.exit(f"frames spread over several directories: {dirs}")
        frame_dir = dirs.pop()
        audio = []
        if info["audio"]:
            # Excerpts re-encode the audio (a stream copy can't cut on our frame boundary).
            acodec = ["-c:a", "aac", "-b:a", "256k"] if (a.start or a.duration) else ["-c:a", "copy"]
            audio = [*cut, "-i", src, *dur]
        cmd = ["ffmpeg", "-hide_banner", "-y", "-framerate", info["fps"], "-pattern_type", "glob",
               "-i", f"{frame_dir}/*.png", *audio, "-map", "0:v:0"]
        if audio:
            cmd += ["-map", "1:a:0", *acodec, "-shortest"]
        cmd += ["-vf", "scale=out_color_matrix=bt709:out_range=tv,format=yuv420p10le",
                "-c:v", "libx265", "-crf", a.crf, "-preset", "slow",
                "-x265-params", "colorprim=bt709:transfer=bt709:colormatrix=bt709:log-level=warning",
                "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709",
                "-color_range", "tv", result]
        run(cmd)

    if "compare" in steps:
        # Left: plain Lanczos of the prepped source. Right: SeedVR2.
        run(["ffmpeg", "-hide_banner", "-y", "-i", prep, "-i", result, "-filter_complex",
             f"[0:v]scale={out_w}:{out_h}:flags=lanczos:in_color_matrix=bt601:out_color_matrix=bt709,"
             f"format=yuv420p,setsar=1[l];"
             f"[1:v]scale=in_color_matrix=bt709:out_color_matrix=bt709,format=yuv420p,setsar=1[r];"
             f"[l][r]hstack=inputs=2[v]", "-map", "[v]",
             "-c:v", "libx264", "-crf", "14", "-preset", "medium", "-colorspace", "bt709",
             "-color_primaries", "bt709", "-color_trc", "bt709", work / "compare.mp4"])
        n = len(list(frames.rglob("*.png")))
        stills = work / "stills"
        stills.mkdir(exist_ok=True)
        fps = float(Fraction(info["fps"]))
        for k, frac in enumerate((0.25, 0.5, 0.75)):
            t = f"{n * frac / fps:.3f}"
            run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-ss", t, "-i", work / "compare.mp4",
                 "-frames:v", "1", stills / f"compare_{k}.png"])
    print(f"done: {work}", flush=True)


if __name__ == "__main__":
    main()
