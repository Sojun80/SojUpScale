#!/usr/bin/env python3
"""Run restoration candidates on evaluation clips and record how each run was made.

  eval/run.py --eval DIR --candidates eval/candidates.json [--clips a,b] [--only x,y] [--gpus 0,1,2,3]

A candidate (see eval/candidates.json) is one of
  {"name": ..., "args": [...]}            upscale.py from this checkout with these arguments
                                          ("gpus": N reserves N GPUs, 0 for CPU-only; default 1)
  {"name": ..., "command": "...", "output": "..."}
                                          any other command (an older checkout, another model's
                                          script); {src} {work} {gpu} {height} are filled in
  {"name": ..., "file": ".../{clip}.mkv"} a finished video made elsewhere (e.g. a commercial
                                          tool's export), copied in for comparison only

Each run goes to DIR/runs/CLIP/CANDIDATE/ with output.mkv and run.json: the
exact command, the candidate definition, git revisions, model hashes, wall
time, frames per second and the GPU (name, memory, peak memory used, sampled
with nvidia-smi). A run whose definition hasn't changed is not redone.
Candidates run in parallel, one GPU each. Stdlib only.
"""
import argparse
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from sojupscale import media  # noqa: E402


def git_rev(path):
    r = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True)
    dirty = subprocess.run(["git", "-C", str(path), "status", "--porcelain", "--untracked-files=no"],
                           capture_output=True, text=True).stdout.strip()
    return (r.stdout.strip() or "unknown") + ("+dirty" if dirty else "")


def gpu_info(gpu):
    r = subprocess.run(["nvidia-smi", "-i", str(gpu), "--query-gpu=name,memory.total,driver_version",
                        "--format=csv,noheader"], capture_output=True, text=True) \
        if shutil.which("nvidia-smi") else None
    return r.stdout.strip() if r and r.returncode == 0 else None


class MemSampler(threading.Thread):
    """Peak memory.used (MiB) of the busiest of the given GPUs, sampled every 2 s."""

    def __init__(self, gpu):
        super().__init__(daemon=True)
        self.gpu, self.peak, self.stop = gpu, None, threading.Event()

    def run(self):
        if not shutil.which("nvidia-smi"):
            return
        while not self.stop.wait(2):
            r = subprocess.run(["nvidia-smi", "-i", str(self.gpu), "--query-gpu=memory.used",
                                "--format=csv,noheader,nounits"], capture_output=True, text=True)
            vals = [int(v) for v in r.stdout.split() if v.strip().isdigit()] if r.returncode == 0 else []
            if vals:
                self.peak = max([self.peak or 0, *vals])


def definition_hash(cand, clip_meta):
    return hashlib.sha256(json.dumps([cand, clip_meta.get("origin_id"), clip_meta["first_frame"],
                                      clip_meta["frames"]], sort_keys=True).encode()).hexdigest()[:16]


def run_one(clip_dir, cand, gpu, a):
    meta = json.loads((clip_dir / "clip.json").read_text())
    out = Path(a.eval) / "runs" / meta["name"] / cand["name"]
    record = out / "run.json"
    h = definition_hash(cand, meta)
    if record.exists() and json.loads(record.read_text()).get("definition") == h and (out / "output.mkv").exists():
        print(f"= {meta['name']}/{cand['name']}: up to date", flush=True)
        return
    out.mkdir(parents=True, exist_ok=True)
    src = clip_dir / "source.mkv"
    work = out / "work"
    fill = dict(src=src, work=work, gpu=gpu, height=a.height, clip=meta["name"], repo=REPO)
    if "file" in cand:
        f = Path(cand["file"].format(**fill))
        shutil.copy(f, out / "output.mkv")
        cmd, produced = None, out / "output.mkv"
    elif "command" in cand:
        cmd = shlex.split(cand["command"].format(**fill))
        produced = Path(cand.get("output", "{work}/upscaled.mkv").format(**fill))
    else:
        cmd = [sys.executable, str(REPO / "upscale.py"), str(src), "--work", str(work),
               "--height", str(a.height), "--compare", "no", "--restart",
               *(["--gpus", str(gpu)] if gpu is not None else []), *cand["args"]]
        produced = work / "upscaled.mkv"
    sampler = MemSampler(gpu) if gpu is not None else None
    t0 = time.time()
    rc = 0
    if cmd:
        print(f"> {meta['name']}/{cand['name']} on {'GPU ' + str(gpu) if gpu is not None else 'CPU'}", flush=True)
        if sampler:
            sampler.start()
        with open(out / "log.txt", "w") as logf:
            rc = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT,
                                env={**os.environ, "PYTHONUNBUFFERED": "1"}).returncode
        if sampler:
            sampler.stop.set()
    secs = time.time() - t0
    if rc != 0 or not produced.exists():
        print(f"! {meta['name']}/{cand['name']} failed (exit {rc}), see {out / 'log.txt'}", flush=True)
        return
    if produced != out / "output.mkv":
        shutil.copy(produced, out / "output.mkv")
    for name in ("plan.json", "timing.json", "manifest.json"):
        if (work / name).exists():
            shutil.copy(work / name, out / name)
    frames = media.frame_count(out / "output.mkv")
    oinfo = media.probe(out / "output.mkv")
    rec = dict(clip=meta["name"], candidate=cand, definition=h, command=cmd, exit_code=rc,
               wall_seconds=round(secs, 1) if cmd else None,
               fps=round(frames / secs, 4) if cmd and secs > 0 else None,
               frames=frames, expected_frames=meta["frames"], size=[oinfo["w"], oinfo["h"]],
               gpu=gpu, gpu_info=gpu_info(gpu) if gpu is not None else None,
               gpu_peak_mib=sampler.peak if sampler else None,
               repo_rev=git_rev(REPO), host=os.uname().nodename,
               finished=time.strftime("%Y-%m-%dT%H:%M:%S"))
    deps = Path(os.environ.get("VIDEOUPSCALE_DEPS", Path.home() / "VideoUpscale-deps"))
    if (deps / "seedvr2").exists():
        rec["upstream_rev"] = git_rev(deps / "seedvr2")
    record.write_text(json.dumps(rec, indent=1))
    if not a.keep_work and work.exists():
        shutil.rmtree(work)
    print(f"< {meta['name']}/{cand['name']}: {frames} frames in {secs:.0f} s", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval", required=True)
    ap.add_argument("--candidates", default=str(REPO / "eval/candidates.json"))
    ap.add_argument("--clips", help="comma-separated clip names (default: all)")
    ap.add_argument("--only", help="comma-separated candidate names (default: all)")
    ap.add_argument("--gpus", default="0,1,2,3", help="GPUs to spread GPU candidates over")
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--keep-work", action="store_true")
    a = ap.parse_args()
    cands = json.loads(Path(a.candidates).read_text())["candidates"]
    if a.only:
        want = a.only.split(",")
        cands = [c for c in cands if c["name"] in want]
    clips = sorted(p.parent for p in (Path(a.eval) / "clips").glob("*/clip.json"))
    if a.clips:
        clips = [c for c in clips if c.name in a.clips.split(",")]
    if not clips or not cands:
        sys.exit("nothing to run (no clips or candidates)")
    tasks = [(c, k) for c in clips for k in cands]
    gpu_free = [g.strip() for g in a.gpus.split(",") if g.strip()]
    lock = threading.Lock()
    threads = []

    def worker(clip, cand, gpu):
        try:
            run_one(clip, cand, gpu, a)
        finally:
            if gpu is not None:
                with lock:
                    gpu_free.extend(gpu.split(","))

    for clip, cand in tasks:
        # "gpus": how many GPUs the candidate gets (0 for CPU-only ones); default 1
        want = 0 if "file" in cand else int(cand.get("gpus", 1))
        if want > len([g for g in a.gpus.split(",") if g.strip()]):
            sys.exit(f"{cand['name']} wants {want} GPUs, only {a.gpus} given")
        gpu = None
        while want:
            with lock:
                if len(gpu_free) >= want:
                    gpu = ",".join(gpu_free[:want])
                    del gpu_free[:want]
                    break
            time.sleep(1)
        t = threading.Thread(target=worker, args=(clip, cand, gpu))
        t.start()
        threads.append(t)
    for t in threads:
        t.join()


if __name__ == "__main__":
    main()
