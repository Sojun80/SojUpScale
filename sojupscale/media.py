"""ffprobe/ffmpeg helpers shared by the orchestrator, the engines and the eval tools.
Stdlib only."""
import json
import re
import subprocess
from fractions import Fraction

# ffprobe colour-space names -> swscale matrix names
_MATRIX = {"bt709": "bt709", "smpte170m": "bt601", "bt470bg": "bt601", "bt601": "bt601",
           "fcc": "fcc", "smpte240m": "smpte240m", "bt2020nc": "bt2020", "bt2020c": "bt2020"}


def probe(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets", "-show_entries",
         "stream=width,height,sample_aspect_ratio,r_frame_rate,nb_read_packets,color_space,color_range,"
         "field_order", "-of", "json", str(path)], check=True, capture_output=True, text=True).stdout
    s = json.loads(out)["streams"][0]
    sar = s.get("sample_aspect_ratio", "1:1")
    sar = Fraction(1) if sar in ("0:1", "N/A", "", None) else Fraction(*map(int, sar.split(":")))
    has_audio = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries",
         "stream=index", "-of", "csv=p=0", str(path)],
        check=True, capture_output=True, text=True).stdout.strip() != ""
    w, h = s["width"], s["height"]
    tagged = _MATRIX.get(s.get("color_space", ""))
    # Untagged: SD is BT.601 (ffmpeg's own default for SD), anything larger BT.709.
    matrix = tagged or ("bt601" if h <= 576 else "bt709")
    rng = "pc" if s.get("color_range") in ("pc", "jpeg") else "tv"
    return dict(w=w, h=h, sar=sar, fps=Fraction(s["r_frame_rate"]), frames=int(s["nb_read_packets"]),
                audio=has_audio, matrix=matrix, matrix_tagged=bool(tagged), range=rng)


def frame_count(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets",
                          "-show_entries", "stream=nb_read_packets", "-of", "csv=p=0", str(path)],
                         capture_output=True, text=True).stdout.strip()
    return int(out) if out.isdigit() else -1


def seek_time(frame, fps):
    # Half a frame early, so the seek lands exactly on `frame` despite timestamp rounding.
    return 0.0 if frame == 0 else float((frame - Fraction(1, 2)) / fps)


def even(x):
    return int(round(x / 2)) * 2


def square_size(w, h, sar):
    """Square-pixel size, never throwing away source pixels."""
    if sar >= 1:
        return even(w * sar), h
    return w, even(h / sar)


def output_size(sq, height):
    """Output size for a short side of `height`, keeping the square-pixel shape.
    Same arithmetic as SeedVR2 (torchvision resize truncates the long side, then
    both sides are floored to even), so every engine produces identical sizes."""
    w, h = sq
    if w >= h:
        return int(height * w / h) // 2 * 2, height // 2 * 2
    return height // 2 * 2, int(height * h / w) // 2 * 2


DEINTERLACE = {
    "none": None,
    # One progressive frame per interlaced frame (frame count unchanged).
    "bwdif": "bwdif=mode=send_frame:parity=auto:deint=all",
    # Telecined film: field matching without decimation (frame count unchanged,
    # the repeated frames stay), leftover combed frames deinterlaced.
    "fieldmatch": "fieldmatch=order=auto:combmatch=full,bwdif=mode=send_frame:deint=interlaced",
}


def to_rgb_filter(info, size, deinterlace="none", flags="lanczos"):
    """Filter chain: source frames -> square pixels at `size` -> RGB48, with the
    source's colour matrix and range applied explicitly (no decoder guesswork)."""
    chain = [DEINTERLACE[deinterlace]] if DEINTERLACE[deinterlace] else []
    chain.append(f"scale={size[0]}:{size[1]}:flags={flags}+accurate_rnd+full_chroma_int+full_chroma_inp:"
                 f"in_color_matrix={info['matrix']}:in_range={info['range']}:out_range=pc")
    chain += ["format=rgb48le", "setsar=1"]
    return ",".join(chain)


def read_cmd(src, info, start, n, size, deinterlace="none", flags="lanczos"):
    """ffmpeg command printing frames [start, start+n) as raw rgb48le on stdout."""
    return ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin",
            "-ss", f"{seek_time(start, info['fps']):.6f}", "-i", str(src), "-frames:v", str(n),
            "-map", "0:v:0", "-an", "-vf", to_rgb_filter(info, size, deinterlace, flags),
            "-f", "rawvideo", "-pix_fmt", "rgb48le", "-"]


def png_cmd(src, info, start, n, size, out_pattern, first_number, deinterlace="none", flags="lanczos",
            depth=16):
    """ffmpeg command writing frames [start, start+n) as PNGs numbered from first_number."""
    return ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
            "-ss", f"{seek_time(start, info['fps']):.6f}", "-i", str(src), "-frames:v", str(n),
            "-map", "0:v:0", "-an", "-vf",
            to_rgb_filter(info, size, deinterlace, flags) + ("" if depth == 16 else ",format=rgb24"),
            "-compression_level", "1", "-start_number", str(first_number), str(out_pattern)]


def encode_cmd(frame_glob, fps, crf, out, preset="slow"):
    """RGB PNGs (8 or 16 bit) -> BT.709 10-bit x265."""
    return ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y", "-framerate", str(fps),
            "-pattern_type", "glob", "-i", str(frame_glob),
            "-vf", "scale=out_color_matrix=bt709:out_range=tv:flags=accurate_rnd,format=yuv420p10le",
            "-c:v", "libx265", "-crf", str(crf), "-preset", preset,
            "-x265-params", "colorprim=bt709:transfer=bt709:colormatrix=bt709:log-level=warning",
            "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709",
            "-color_range", "tv", str(out)]


def interlace_check(src, start):
    """Fraction of frames idet calls interlaced, over 30 s from `start`."""
    r = subprocess.run(
        ["ffmpeg", "-hide_banner", "-nostats", "-nostdin", "-ss", str(start), "-i", str(src), "-t", "30",
         "-an", "-vf", "idet", "-f", "null", "-"], capture_output=True, text=True)
    for line in r.stderr.splitlines():
        if "Multi frame detection" in line:
            nums = [int(t) for t in line.split() if t.isdigit()]
            tff, bff, prog = nums[0], nums[1], nums[2]
            return (tff + bff) / max(1, tff + bff + prog)
    return 0.0


def scene_cuts(src, fps, first, last, threshold):
    """Frame numbers (absolute) where a new shot starts, within (first, last)."""
    # The frame counter, not the timestamp: after a seek the timestamps are offset by
    # half a frame, and rounding them put cuts one frame late.
    r = subprocess.run(
        ["ffmpeg", "-hide_banner", "-nostats", "-nostdin", "-ss", f"{seek_time(first, fps):.6f}",
         "-i", str(src), "-frames:v", str(last - first), "-an",
         "-vf", f"scdet=threshold={threshold},metadata=mode=print:key=lavfi.scd.time",
         "-f", "null", "-"], capture_output=True, text=True, check=True)
    frames = [int(n) for n in re.findall(r"frame:(\d+)\s+pts:", r.stderr)]
    return sorted({first + n for n in frames} - {first})
