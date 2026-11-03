#!/usr/bin/env python3
"""Make evaluation clips: exact, losslessly stored copies of short source ranges.

  eval/clips.py add SRC --eval DIR --name faces1 --start 812.5 --duration 8 --tags faces,motion
  eval/clips.py add HD.mkv --eval DIR --name hd-texture --start 60 --duration 8 --degrade dvd
  eval/clips.py list --eval DIR

Each clip lives in DIR/clips/NAME/:
  source.mkv   the input every candidate gets: the exact frame range, decoded
               and stored losslessly (FFV1) with the source's size, SAR and
               colour tags, so all candidates see identical pixels
  reference.mkv  only with --degrade: the original before degradation, at the
               degraded clip's display shape, for full-reference metrics
  clip.json    where it came from (file identity, frame range), tags, notes

--degrade makes ground-truth clips from good footage: the original is shrunk
to SD and compressed like old video, so you can see what a model recovers and
what it invents. Real old footage (no reference) is still the main test.
Stdlib only.
"""
import argparse
import json
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from sojupscale import manifest, media  # noqa: E402

# name -> function(display aspect) -> (filter applied to the original, encoder args).
# Each keeps the original's display aspect (anamorphic where SD video was).
def _dvd(dar):
    return (f"scale=720:480:flags=bicubic,setsar={dar * 480 / 720}",
            ["-c:v", "mpeg2video", "-b:v", "2500k", "-maxrate", "4000k", "-bufsize", "1835k", "-g", "15"])


def _web360(dar):
    return (f"scale={media.even(360 * dar)}:360:flags=bicubic,setsar=1",
            ["-c:v", "libx264", "-crf", "32", "-preset", "medium"])


def _vhs(dar):
    w = media.even(480 * dar)
    return (f"scale={media.even(w * 0.45)}:480:flags=bicubic,scale={w}:480:flags=bicubic,"
            f"gblur=sigma=0.8,noise=alls=8:allf=t,setsar=1",
            ["-c:v", "libx264", "-crf", "26", "-preset", "medium"])


DEGRADE = {
    "dvd": _dvd,        # DVD-ish: 720x480 anamorphic, MPEG-2 at a starved bitrate
    "web360": _web360,  # old web video: 360p, heavy H.264
    "vhs": _vhs,        # VHS-ish: low horizontal resolution, soft, noisy
}


def ffmpeg(*args):
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y", *map(str, args)]
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def add(a):
    src = Path(a.src).resolve()
    info = media.probe(src)
    fps = info["fps"]
    first = round(a.start * fps)
    n = min(round(a.duration * fps), info["frames"] - first)
    if n <= 0:
        sys.exit("empty range")
    d = Path(a.eval) / "clips" / a.name
    if d.exists() and not a.force:
        sys.exit(f"{d} exists (use --force to replace)")
    d.mkdir(parents=True, exist_ok=True)
    seek = ["-ss", f"{media.seek_time(first, fps):.6f}", "-i", src, "-frames:v", n, "-map", "0:v:0"]
    if a.degrade:
        sq = media.square_size(info["w"], info["h"], info["sar"])
        vf, enc = DEGRADE[a.degrade](Fraction(sq[0], sq[1]))
        degraded = d / "degraded_tmp.mkv"
        # Degrade from the original pixels, then store the degraded result losslessly.
        ffmpeg(*seek, "-vf", vf + ",format=yuv420p", *enc, "-an", degraded)
        ffmpeg("-i", degraded, "-map", "0:v:0", "-c:v", "ffv1", "-level", "3",
               "-color_primaries", "smpte170m", "-color_trc", "smpte170m", "-colorspace", "smpte170m",
               "-color_range", "tv", d / "source.mkv")
        degraded.unlink()
        # Reference: the original at its square-pixel display shape, short side at most 1080.
        ref_h = min(sq[1], 1080)
        ffmpeg(*seek, "-vf", media.to_rgb_filter(info, (media.even(ref_h * sq[0] / sq[1]), ref_h)),
               "-c:v", "ffv1", "-level", "3", "-pix_fmt", "rgb48le", "-an", d / "reference.mkv")
    else:
        # Exact decoded pixels, losslessly; pixel format, SAR and colour tags carry over.
        ffmpeg(*seek, "-map", "0:a:0?", "-t", f"{float(n / fps):.6f}", "-c:a", "flac",
               "-c:v", "ffv1", "-level", "3", d / "source.mkv")
    got = media.frame_count(d / "source.mkv")
    if got != n:
        sys.exit(f"{d}/source.mkv has {got} frames, expected {n}")
    meta = dict(name=a.name, origin=str(src), origin_id=manifest.file_id(src), first_frame=first,
                frames=n, fps=str(fps), tags=[t for t in a.tags.split(",") if t], notes=a.notes,
                degrade=a.degrade, crop=a.crop, row=a.row)
    (d / "clip.json").write_text(json.dumps(meta, indent=1))
    print(f"clip {a.name}: {n} frames from {src.name} at frame {first}", flush=True)


def list_clips(a):
    for f in sorted((Path(a.eval) / "clips").glob("*/clip.json")):
        m = json.loads(f.read_text())
        print(f"{m['name']:20} {m['frames']:5} frames  tags={','.join(m['tags'])}  "
              f"{'degraded:' + m['degrade'] if m['degrade'] else 'real'}  {m['notes']}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("add")
    p.add_argument("src")
    p.add_argument("--eval", required=True, help="evaluation directory (keep it outside git)")
    p.add_argument("--name", required=True)
    p.add_argument("--start", type=float, required=True, help="seconds")
    p.add_argument("--duration", type=float, default=8, help="seconds (5-10 is plenty)")
    p.add_argument("--tags", default="", help="e.g. faces,motion,texture,compression,dark")
    p.add_argument("--notes", default="")
    p.add_argument("--degrade", choices=sorted(DEGRADE), help="make a ground-truth clip from good footage")
    p.add_argument("--crop", help="x,y,w,h region of interest in output pixels, zoomed in the report")
    p.add_argument("--row", type=float, default=0.5, help="row (0-1) used for the x-t temporal slice")
    p.add_argument("--force", action="store_true")
    p.set_defaults(fn=add)
    p = sub.add_parser("list")
    p.add_argument("--eval", required=True)
    p.set_defaults(fn=list_clips)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
