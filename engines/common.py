"""Job queue protocol shared by all engine workers. Stdlib only.

The orchestrator (upscale.py) writes one JSON file per job into WORK/queue/ and
starts one worker per GPU. A worker claims a job by renaming its file into
WORK/running/ (atomic, so two workers never take the same job), processes the
pieces, writes JOB/done and takes the next job until the queue is empty. The
model therefore loads once per GPU per run, and fast GPUs simply take more jobs.

A job's frames go to JOB/frames/fNNNNNNN.png (absolute source frame numbers).
Each finished piece leaves JOB/pieces/pNNN.json, so an interrupted job resumes
at the next unfinished piece.

To add an engine: write engines/<name>_runner.py whose main() calls
run_worker(process_job), and register it in upscale.py's ENGINES.
"""
import argparse
import json
import os
import sys
import time
import traceback
from fractions import Fraction
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def frame_name(n):
    return f"f{n:07d}.png"


def source_info(job):
    info = dict(job["source"]["info"])
    info["fps"] = Fraction(info["fps"])
    return info


def piece_marker(job, i):
    return Path(job["dir"]) / "pieces" / f"p{i:03d}.json"


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1))
    os.replace(tmp, path)


def claim(queue, running, worker):
    """Next job file from the queue, moved to running/; None when empty."""
    for f in sorted(Path(queue).glob("*.json")):
        dest = Path(running) / f"{f.stem}.{worker}.json"
        try:
            os.rename(f, dest)
        except FileNotFoundError:
            continue  # another worker took it
        return dest
    return None


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run_worker(process_job, setup=None):
    """Worker main loop. process_job(job, state) processes all unfinished pieces;
    setup() runs once, before the first job, and returns `state` (e.g. a loaded
    model)."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--worker", required=True)
    a = ap.parse_args()
    work = Path(a.work)
    queue, running = work / "queue", work / "running"
    running.mkdir(exist_ok=True)
    state = None
    while True:
        if (work / "cancel").exists():
            log("cancel requested")
            sys.exit(3)
        claimed = claim(queue, running, a.worker)
        if claimed is None:
            break
        job = json.loads(claimed.read_text())
        log(f"{a.worker}: job {job['id']} ({len(job['pieces'])} pieces)")
        try:
            if state is None and setup is not None:
                state = setup(job)
            t0 = time.time()
            process_job(job, state)
            write_json(Path(job["dir"]) / "done", dict(worker=a.worker, seconds=round(time.time() - t0, 1)))
            claimed.unlink()
        except BaseException:
            traceback.print_exc()
            os.replace(claimed, queue / f"{job['id']}.json")  # hand it back for the next run
            sys.exit(1)
