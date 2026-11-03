import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
HAVE_FFMPEG = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


def make_clip(path, seconds=(2, 5, 1.5), size="720x480", sar="8/9", audio=True, rate="30000/1001"):
    """Shots of different synthetic content joined by hard cuts; the middle one
    (mandelbrot zoom) moves all the time. Returns the cut frame numbers."""
    w, h = size.split("x")
    srcs = [f"testsrc2=s={size}:r={rate}:d={seconds[0]}",
            f"mandelbrot=s={size}:r={rate}",
            f"smptebars=s={size}:r={rate}:d={seconds[2]}"]
    fps = eval(rate.replace("/", "/"))
    n_mid = round(seconds[1] * fps)
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"]
    for s in srcs:
        cmd += ["-f", "lavfi", "-i", s]
    fc = f"[1]trim=end_frame={n_mid},setpts=PTS-STARTPTS[m];[0][m][2]concat=n=3:v=1,setsar={sar}[v]"
    maps = ["-map", "[v]"]
    if audio:
        cmd += ["-f", "lavfi", "-i", f"sine=f=440:d={sum(seconds) + 1}"]
        maps += ["-map", "3:a", "-c:a", "aac", "-shortest"]
    cmd += ["-filter_complex", fc, *maps, "-c:v", "libx264", "-crf", "20", "-bf", "3", "-g", "30",
            "-pix_fmt", "yuv420p", str(path)]
    subprocess.run(cmd, check=True)
    n0 = round(seconds[0] * fps)
    return [n0, n0 + n_mid]


def upscale(*args, check=True):
    return subprocess.run([sys.executable, str(REPO / "upscale.py"), *map(str, args)],
                          capture_output=True, text=True, check=False)


def per_frame_psnr(a, b, filter_a="", filter_b=""):
    """PSNR (Y) per frame between two videos, via ffmpeg's psnr filter."""
    with tempfile.TemporaryDirectory() as d:
        stats = Path(d) / "psnr.log"
        fa = (filter_a + ",") if filter_a else ""
        fb = (filter_b + ",") if filter_b else ""
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(a), "-i", str(b),
                        "-filter_complex", f"[0:v]{fa}format=yuv420p[a];[1:v]{fb}format=yuv420p[b];"
                        f"[a][b]psnr=stats_file={stats}", "-f", "null", "-"], check=True)
        vals = []
        for line in stats.read_text().splitlines():
            m = re.search(r"psnr_y:(\S+)", line)
            vals.append(float("inf") if m.group(1) == "inf" else float(m.group(1)))
        return vals
