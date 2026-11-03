#!/usr/bin/env python3
"""Reference engine: plain Lanczos resize with ffmpeg (CPU, stdlib only).

The baseline every restoration candidate is compared against, and a stand-in
for the GPU engine when testing the pipeline without a GPU. A per-frame resize
needs no crossfades: overlapping pieces produce identical frames, so each piece
simply writes its whole keep range."""
import subprocess
import time
from pathlib import Path

from common import frame_name, log, piece_marker, run_worker, source_info, write_json

from sojupscale import media


def process_job(job, _state):
    info = source_info(job)
    frames = Path(job["dir"]) / "frames"
    frames.mkdir(parents=True, exist_ok=True)
    depth = job["output"]["png_depth"]
    for i, p in enumerate(job["pieces"]):
        marker = piece_marker(job, i)
        if marker.exists():
            continue
        t0 = time.time()
        # Read from the piece start (deinterlacers want neighbouring frames), keep the keep range.
        cmd = media.png_cmd(job["source"]["path"], info, p["start"], p["end"] - p["start"],
                            job["output"]["size"], frames / "f%07d.png", p["keep_start"],
                            job["source"]["deinterlace"], flags="lanczos", depth=depth)
        vf = cmd.index("-vf") + 1
        cmd[vf] += f",trim=start_frame={p['keep_start'] - p['start']}:end_frame={p['keep_end'] - p['start']},"
        cmd[vf] += "setpts=PTS-STARTPTS"  # else the image muxer pads the trimmed-off start with duplicates
        subprocess.run(cmd, check=True)
        missing = [n for n in range(p["keep_start"], p["keep_end"]) if not (frames / frame_name(n)).exists()]
        if missing:
            raise RuntimeError(f"lanczos: piece {i} is missing {len(missing)} frames, first {missing[0]}")
        write_json(marker, dict(frames=p["keep_end"] - p["keep_start"], seconds=round(time.time() - t0, 2)))
        log(f"piece {i}: frames {p['keep_start']}-{p['keep_end']}")


if __name__ == "__main__":
    run_worker(process_job)
