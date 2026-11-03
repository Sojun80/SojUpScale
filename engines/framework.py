"""Piece processing for model engines (numpy + OpenCV, both in the deps venv).

process_job(job, restore) reads each piece's frames straight from the source
with ffmpeg (exact frame range, square pixels, explicit colour matrix, 16-bit
RGB), calls restore(frames, piece), crossfades with the previous piece where the
plan says so, and writes rounded 16-bit (or 8-bit) PNGs in parallel.

restore(frames, piece) gets float32 [T, H, W, 3] in [0, 1] at the square-pixel
source size and returns T frames at the job's output size, values in [0, 1], as
a numpy array or a CPU torch tensor."""
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from common import frame_name, log, piece_marker, source_info, write_json

from sojupscale import media

PNG_THREADS = 16


def read_frames(job, piece):
    info = source_info(job)
    w, h = job["source"]["square"]
    n = piece["end"] - piece["start"]
    cmd = media.read_cmd(job["source"]["path"], info, piece["start"], n, (w, h), job["source"]["deinterlace"])
    raw = subprocess.run(cmd, check=True, capture_output=True).stdout
    got = len(raw) // (w * h * 6)
    if got != n or len(raw) % (w * h * 6):
        raise RuntimeError(f"read {got} frames for piece {piece['start']}-{piece['end']}, expected {n}")
    x = np.frombuffer(raw, np.uint16).reshape(n, h, w, 3)
    return x.astype(np.float32) * (1.0 / 65535.0)


def blend_weights(k):
    """Weight of the previous piece over a k-frame overlap: the same curve as
    SeedVR2's batch overlap (Hann crossfade over the middle third, so the
    outermost frames of each side, which have the least temporal context, are
    not used)."""
    if k <= 0:
        return np.zeros(0, np.float32)
    if k < 3:
        return np.linspace(1.0, 0.0, k, dtype=np.float32)
    t = np.linspace(0.0, 1.0, k, dtype=np.float32)
    u = np.clip((t - 1 / 3) * 3, 0.0, 1.0)
    return (0.5 + 0.5 * np.cos(np.pi * u)).astype(np.float32)


def _as_np(frame):
    return frame.numpy() if hasattr(frame, "numpy") else np.asarray(frame)


def to_png_array(frame, depth):
    """float [0,1] RGB -> rounded uint16/uint8 BGR (upstream truncated to 8 bits,
    which also darkened everything by half a step on average)."""
    peak = 65535.0 if depth == 16 else 255.0
    a = np.clip(frame * peak + 0.5, 0, peak).astype(np.uint16 if depth == 16 else np.uint8)
    return cv2.cvtColor(a, cv2.COLOR_RGB2BGR)


def read_png(path):
    a = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if a is None:
        raise RuntimeError(f"cannot read {path}")
    peak = 65535.0 if a.dtype == np.uint16 else 255.0
    return cv2.cvtColor(a, cv2.COLOR_BGR2RGB).astype(np.float32) / peak


def write_png(path, arr):
    tmp = path.with_name(path.stem + ".tmp.png")
    if not cv2.imwrite(str(tmp), arr, [cv2.IMWRITE_PNG_COMPRESSION, 1]):
        raise RuntimeError(f"cannot write {path}")
    tmp.replace(path)


def process_job(job, restore):
    d = Path(job["dir"])
    frames, tail = d / "frames", d / "tail"
    frames.mkdir(parents=True, exist_ok=True)
    tail.mkdir(exist_ok=True)
    depth = job["output"]["png_depth"]
    out_w, out_h = job["output"]["size"]
    for i, p in enumerate(job["pieces"]):
        marker = piece_marker(job, i)
        if marker.exists():
            continue
        t0 = time.time()
        x = read_frames(job, p)
        t1 = time.time()
        y = restore(x, p)
        t2 = time.time()
        del x
        if len(y) != p["end"] - p["start"] or tuple(y[0].shape) != (out_h, out_w, 3):
            raise RuntimeError(f"engine returned {len(y)} frames of {tuple(y[0].shape)}, expected "
                               f"{p['end'] - p['start']} of {(out_h, out_w, 3)}")
        w_prev = blend_weights(p["blend_in"])
        held_from = p["keep_end"] - p["blend_out"]

        def save(n):
            frame = _as_np(y[n - p["start"]]).astype(np.float32, copy=False)
            j = n - p["keep_start"]
            if j < p["blend_in"]:
                frame = read_png(tail / frame_name(n)) * w_prev[j] + frame * (1.0 - w_prev[j])
            write_png((tail if n >= held_from else frames) / frame_name(n), to_png_array(frame, depth))

        with ThreadPoolExecutor(PNG_THREADS) as pool:
            list(pool.map(save, range(p["keep_start"], p["keep_end"])))
        stats = dict(frames=p["keep_end"] - p["keep_start"], read_s=round(t1 - t0, 2),
                     restore_s=round(t2 - t1, 2), write_s=round(time.time() - t2, 2))
        stats.update(getattr(restore, "last_stats", {}))
        write_json(marker, stats)
        for n in range(p["keep_start"], p["keep_start"] + p["blend_in"]):
            (tail / frame_name(n)).unlink(missing_ok=True)  # consumed
        log(f"piece {i + 1}/{len(job['pieces'])}: frames {p['start']}-{p['end']} "
            f"(keep {p['keep_start']}-{p['keep_end']}) restore {t2 - t1:.1f} s "
            f"= {(p['end'] - p['start']) / max(t2 - t1, 1e-9):.3f} fps")
        del y
