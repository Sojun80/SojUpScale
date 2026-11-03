#!/usr/bin/env python3
"""Restore and upscale old SD video (SeedVR2 by default).

Planning (CPU, before any GPU work):
  * scene cuts split the range into shots; the model never sees across a cut
  * shots longer than --max-piece-seconds are split into overlapping pieces
    that are crossfaded (or, between jobs, cut mid-overlap with context)
  * consecutive shots are grouped into jobs of about --segment-seconds
Processing:
  * one worker per GPU loads the model once and takes jobs from a queue
    (WORK/queue); each job's frames are written as 16-bit PNGs
  * finished jobs are encoded to BT.709 10-bit x265 on the CPU while the GPUs
    continue, and their PNGs are deleted
  * the jobs are joined, the source audio muxed back, and for short ranges a
    Lanczos-vs-restored side-by-side is made

Resume by rerunning with the same --work. WORK/manifest.json records the source
content, range and every setting that affects the pixels; a rerun with anything
different is refused instead of silently mixing results (--restart discards).

Interlaced or telecined sources are refused unless --deinterlace is given (or
--allow-interlaced): combed frames fed to the model get the combing baked in.

Stdlib only; engines run in their own interpreter (SeedVR2: the deps venv from
thed1/setup.sh).
"""
import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
from sojupscale import manifest, media, plan  # noqa: E402

DEPS = Path(os.environ.get("VIDEOUPSCALE_DEPS", Path.home() / "VideoUpscale-deps"))
COMPARE_MAX_SECONDS = 120


def seedvr2_version():
    m = re.search(r"^ENGINE_VERSION = (\d+)", (REPO / "engines/seedvr2_runner.py").read_text(), re.M)
    rev = subprocess.run(["git", "-C", str(DEPS / "seedvr2"), "rev-parse", "HEAD"],
                         capture_output=True, text=True).stdout.strip() or "missing"
    return dict(runner=int(m.group(1)), upstream=rev,
                patches=manifest.tree_hash((REPO / "thed1/patches").glob("*.patch")))


def model_id(name):
    """sha256 from thed1/models.txt (setup.sh verifies downloads against it)."""
    for line in (REPO / "thed1/models.txt").read_text().splitlines():
        parts = line.split()
        if parts and parts[0] == name:
            return parts[2]
    p = DEPS / "models/SEEDVR2" / name
    return f"size:{p.stat().st_size}" if p.exists() else "unknown"


ENGINES = {
    "seedvr2": dict(python=lambda: DEPS / "venv/bin/python", runner="seedvr2_runner.py", gpu=True),
    "lanczos": dict(python=lambda: sys.executable, runner="lanczos_runner.py", gpu=False),
}


def engine_settings(a):
    if a.engine == "seedvr2":
        return dict(model=a.model, model_sha256=model_id(a.model), batch_size=a.batch_size,
                    overlap=a.overlap, balanced_batches=not a.no_balanced_batches,
                    color_correction=a.color_correction, attention=a.attention, seed=a.seed,
                    prepend_frames=a.prepend_frames, input_noise=a.input_noise,
                    latent_noise=a.latent_noise, compile=a.compile, version=seedvr2_version())
    return dict(version=subprocess.run(["ffmpeg", "-version"], capture_output=True,
                                       text=True).stdout.split("\n")[0])


# Settings that only change speed or logging, kept out of the resume identity.
NON_PIXEL_SETTINGS = {"compile"}


def show(cmd):
    print("+ " + " ".join(str(c) for c in cmd), flush=True)
    return [str(c) for c in cmd]


def run(cmd):
    subprocess.run(show(cmd), check=True)


def write_json(path, obj):
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1))
    os.replace(tmp, path)


WORK_ITEMS = ["manifest.json", "plan.json", "queue", "running", "jobs", "workers", "progress.json",
              "timing.json", "video.mkv", "upscaled.mkv", "compare.mp4", "segments.txt", "cancel",
              "segments.json"]


def restart(work):
    """Delete what this script creates in WORK (never anything else)."""
    for name in WORK_ITEMS:
        p = work / name
        if p.is_dir():
            shutil.rmtree(p)
        elif p.exists():
            p.unlink()
    for p in work.glob("seg[0-9][0-9][0-9]"):  # older versions' segments
        shutil.rmtree(p)


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src")
    ap.add_argument("--work", required=True, help="work/output directory (reuse it to resume)")
    ap.add_argument("--start", type=float, default=0, help="range start, seconds")
    ap.add_argument("--duration", type=float, default=0, help="range length, seconds (0 = to end)")
    ap.add_argument("--height", type=int, default=1080, help="output height (short side)")
    ap.add_argument("--engine", choices=sorted(ENGINES), default="seedvr2")
    g = ap.add_argument_group("planning")
    g.add_argument("--segment-seconds", type=float, default=20,
                   help="job length: frames one GPU processes and encodes in one go")
    g.add_argument("--max-piece-seconds", type=float, default=20,
                   help="longest piece per model call (RAM: ~20 GB per GPU worker for 20 s at 1080p)")
    g.add_argument("--piece-overlap", type=int, default=12,
                   help="frames shared by consecutive pieces of a long shot (crossfaded)")
    g.add_argument("--scene-threshold", type=float, default=10, help="ffmpeg scdet score that counts as a cut")
    g.add_argument("--deinterlace", choices=sorted(media.DEINTERLACE), default="none")
    g.add_argument("--allow-interlaced", action="store_true", help="process interlaced input as is")
    g = ap.add_argument_group("SeedVR2")
    g.add_argument("--model", default="seedvr2_ema_7b_fp16.safetensors")
    g.add_argument("--gpus", default="0,1,2,3", help="CUDA devices, one worker each, e.g. 2 or 0,1,2,3")
    g.add_argument("--batch-size", type=int, default=45, help="max frames per batch, 4n+1")
    g.add_argument("--overlap", type=int, default=3, help="frames shared by consecutive batches")
    g.add_argument("--no-balanced-batches", action="store_true",
                   help="use --batch-size as is (short last batch per piece), like upstream")
    g.add_argument("--color-correction", default="lab",
                   choices=["lab", "wavelet", "wavelet_adaptive", "hsv", "adain", "none"])
    g.add_argument("--attention", default="sdpa")
    g.add_argument("--seed", type=int, default=42)
    g.add_argument("--prepend-frames", type=int, default=0, help="mirrored frames before each piece")
    g.add_argument("--input-noise", type=float, default=0.0, help="upstream input_noise_scale")
    g.add_argument("--latent-noise", type=float, default=0.0, help="upstream latent_noise_scale")
    g.add_argument("--compile", action="store_true", help="torch.compile DiT and VAE (speed only)")
    g = ap.add_argument_group("output")
    g.add_argument("--png-depth", type=int, choices=[8, 16], default=16, help="intermediate frame depth")
    g.add_argument("--crf", type=int, default=16)
    g.add_argument("--x265-preset", default="slow")
    g.add_argument("--encoders", type=int, default=1, help="x265 encodes running at once")
    g.add_argument("--workers", type=int, default=1, help="workers for CPU engines")
    g.add_argument("--compare", choices=["auto", "yes", "no"], default="auto",
                   help=f"side-by-side video (auto: ranges up to {COMPARE_MAX_SECONDS} s)")
    g.add_argument("--keep-frames", action="store_true", help="keep each job's PNGs")
    g.add_argument("--restart", action="store_true", help="discard this work directory's results first")
    a = ap.parse_args(argv)
    if (a.batch_size - 1) % 4:
        ap.error("--batch-size must be 4n+1 (1, 5, 9, 13, 17, 21, ...)")
    if not 0 <= a.overlap < a.batch_size:
        ap.error("--overlap must be smaller than --batch-size")
    return a


class Abort(Exception):
    pass


class Workers:
    """Engine worker processes, their output streamed live with a prefix and
    copied to WORK/workers/<name>.log."""

    def __init__(self):
        self.procs = {}

    def start(self, name, cmd, env, log_path):
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                bufsize=1, env=env)

        def pump():
            with open(log_path, "a") as f:
                for line in proc.stdout:
                    sys.stdout.write(f"[{name}] {line}")
                    f.write(line)
                    f.flush()

        t = threading.Thread(target=pump, daemon=True)
        t.start()
        self.procs[name] = (proc, t)

    def poll(self):
        """Names of workers that exited with an error; finished ones are dropped."""
        failed = []
        for name, (proc, t) in list(self.procs.items()):
            rc = proc.poll()
            if rc is not None:
                t.join(timeout=5)
                del self.procs[name]
                if rc != 0:
                    failed.append(f"{name} (exit {rc})")
        return failed

    def stop(self):
        for proc, _ in self.procs.values():
            if proc.poll() is None:
                proc.terminate()
        for proc, _ in self.procs.values():
            try:
                proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                proc.kill()


def job_spec(i, job, work, src, info, sq, out, a, settings):
    return dict(id=f"j{i:03d}", dir=str(work / "jobs" / f"j{i:03d}"), pieces=job,
                source=dict(path=str(src), square=list(sq), deinterlace=a.deinterlace,
                            info=dict(w=info["w"], h=info["h"], fps=str(info["fps"]),
                                      matrix=info["matrix"], range=info["range"])),
                output=dict(size=list(out), png_depth=a.png_depth),
                engine=dict(name=a.engine, settings=settings))


def progress(work, jobs, state, t0, frames_total, extra=None):
    done = 0
    for i in range(len(jobs)):
        d = work / "jobs" / f"j{i:03d}"
        if (d / "encoded").exists():
            done += sum(len(plan.written_frames(p)) for p in jobs[i])
            continue
        for m in (d / "pieces").glob("p*.json") if (d / "pieces").exists() else []:
            try:
                done += json.loads(m.read_text())["frames"]
            except (ValueError, KeyError, OSError):
                pass
    elapsed = time.time() - t0
    p = dict(state=state, frames_total=frames_total, frames_done=min(done, frames_total),
             fraction=round(min(done, frames_total) / max(1, frames_total), 4), elapsed_s=round(elapsed),
             updated=time.strftime("%Y-%m-%dT%H:%M:%S"), **(extra or {}))
    write_json(work / "progress.json", p)
    return p


def main(argv=None):
    a = parse_args(argv)
    src = Path(a.src).resolve()
    work = Path(a.work).expanduser().resolve()
    work.mkdir(parents=True, exist_ok=True)
    if a.restart:
        restart(work)
    (work / "cancel").unlink(missing_ok=True)
    eng = ENGINES[a.engine]

    info = media.probe(src)
    fps = info["fps"]
    sq = media.square_size(info["w"], info["h"], info["sar"])
    out = media.output_size(sq, a.height)
    first = round(a.start * fps)
    last = min(info["frames"], first + round(a.duration * fps)) if a.duration else info["frames"]
    if last <= first:
        sys.exit("empty range")
    job_frames = max(1, round(a.segment_seconds * fps))
    max_piece = max(1, round(a.max_piece_seconds * fps))
    if job_frames < max_piece:
        sys.exit("--segment-seconds must be >= --max-piece-seconds")
    if not 0 <= a.piece_overlap <= max_piece // 4:
        sys.exit("--piece-overlap must be between 0 and a quarter of a piece")
    print(f"source {info['w']}x{info['h']} SAR {info['sar']} @ {fps} ({info['frames']} frames, "
          f"{info['matrix']}{'' if info['matrix_tagged'] else ' assumed'}, {info['range']} range) -> "
          f"square {sq[0]}x{sq[1]} -> output {out[0]}x{out[1]}; range frames {first}-{last}", flush=True)

    settings = engine_settings(a)
    identity = dict(
        source=manifest.file_id(src), range=[first, last], square=list(sq), output=list(out),
        matrix=info["matrix"], color_range=info["range"], deinterlace=a.deinterlace,
        plan=dict(job_frames=job_frames, max_piece=max_piece, piece_overlap=a.piece_overlap,
                  scene_threshold=a.scene_threshold),
        engine=a.engine, settings={k: v for k, v in settings.items() if k not in NON_PIXEL_SETTINGS},
        png_depth=a.png_depth, encode=dict(crf=a.crf, preset=a.x265_preset))
    try:
        manifest.ensure(work, identity)
    except manifest.Mismatch as e:
        sys.exit(str(e))

    plan_file = work / "plan.json"
    if plan_file.exists():
        p = json.loads(plan_file.read_text())
        jobs, cuts = p["jobs"], p["scene_cuts"]
    else:
        if a.deinterlace == "none" and not a.allow_interlaced:
            ratio = media.interlace_check(src, a.start)
            print(f"idet: {ratio:.1%} of frames interlaced", flush=True)
            if ratio > 0.05:
                sys.exit("source looks interlaced/telecined; use --deinterlace bwdif (video) or "
                         "--deinterlace fieldmatch (telecined film), or --allow-interlaced")
        cuts = media.scene_cuts(src, fps, first, last, a.scene_threshold)
        jobs = plan.plan_jobs(first, last, cuts, job_frames, max_piece, a.piece_overlap)
        plan.check(jobs, first, last, cuts)
        write_json(plan_file, dict(first=first, last=last, scene_cuts=cuts, jobs=jobs))
    n_pieces = sum(len(j) for j in jobs)
    print(f"{len(cuts)} scene cuts, {n_pieces} pieces in {len(jobs)} jobs", flush=True)

    # Queue every job that isn't finished; jobs a crashed run left in running/ go back too.
    for d in ("queue", "running", "workers", "jobs"):
        (work / d).mkdir(exist_ok=True)
    for f in (work / "running").glob("*.json"):
        os.replace(f, work / "queue" / (f.name.split(".")[0] + ".json"))
    to_encode = []
    for i, job in enumerate(jobs):
        d = work / "jobs" / f"j{i:03d}"
        qf = work / "queue" / f"j{i:03d}.json"
        if (d / "encoded").exists():
            qf.unlink(missing_ok=True)
        elif (d / "done").exists():
            qf.unlink(missing_ok=True)
            to_encode.append(i)
        else:
            write_json(qf, job_spec(i, job, work, src, info, sq, out, a, settings))

    t0 = time.time()
    workers = Workers()
    encodes = {}  # job index -> Popen
    stop = {"why": None}

    def on_signal(signum, _frame):
        stop["why"] = f"signal {signum}"

    old_handlers = {s: signal.signal(s, on_signal) for s in (signal.SIGINT, signal.SIGTERM)}

    if any((work / "queue").glob("*.json")):
        env = {**os.environ, "PYTHONUNBUFFERED": "1", "VIDEOUPSCALE_DEPS": str(DEPS)}
        names = ([f"gpu{g.strip()}" for g in a.gpus.split(",") if g.strip()] if eng["gpu"]
                 else [f"cpu{k}" for k in range(a.workers)])
        python = str(eng["python"]())
        for name in names:
            wenv = dict(env)
            if eng["gpu"]:
                wenv["CUDA_VISIBLE_DEVICES"] = name[3:]
            workers.start(name, show([python, REPO / "engines" / eng["runner"], "--work", work,
                                      "--worker", name]), wenv, work / "workers" / f"{name}.log")

    def encode_done(i, proc):
        d = work / "jobs" / f"j{i:03d}"
        if proc.returncode != 0:
            raise Abort(f"x265 encode failed for {d}")
        want = sum(len(plan.written_frames(p)) for p in jobs[i])
        got = media.frame_count(d / "video.mkv")
        if got != want:
            raise Abort(f"{d}/video.mkv has {got} frames, expected {want}")
        (d / "encoded").write_text("ok\n")
        if not a.keep_frames:
            shutil.rmtree(d / "frames", ignore_errors=True)
            shutil.rmtree(d / "tail", ignore_errors=True)
        print(f"job j{i:03d} encoded ({got} frames)", flush=True)

    frames_total = last - first
    last_progress = 0
    try:
        while True:
            if (work / "cancel").exists():
                stop["why"] = "cancel file"
            if stop["why"]:
                raise Abort(f"stopped ({stop['why']}); rerun the same command to resume")
            failed = workers.poll()
            if failed:
                raise Abort(f"worker failed: {', '.join(failed)}; see {work / 'workers'}")
            for i in range(len(jobs)):
                d = work / "jobs" / f"j{i:03d}"
                if (d / "done").exists() and not (d / "encoded").exists() and i not in encodes \
                        and i not in to_encode:
                    to_encode.append(i)
            for i, proc in list(encodes.items()):
                if proc.poll() is not None:
                    del encodes[i]
                    encode_done(i, proc)
            to_encode.sort()
            while to_encode and len(encodes) < a.encoders:
                i = to_encode.pop(0)
                d = work / "jobs" / f"j{i:03d}"
                encodes[i] = subprocess.Popen(show(media.encode_cmd(d / "frames" / "f*.png", fps, a.crf,
                                                                    d / "video.mkv", a.x265_preset)))
            if all((work / "jobs" / f"j{i:03d}" / "encoded").exists() for i in range(len(jobs))):
                break
            if not workers.procs and not encodes and not to_encode:
                pending = [i for i in range(len(jobs)) if not (work / "jobs" / f"j{i:03d}" / "done").exists()]
                if pending:
                    raise Abort(f"workers finished but jobs {pending} are not done")
            if time.time() - last_progress > 5:
                progress(work, jobs, "processing", t0, frames_total,
                         dict(jobs_total=len(jobs), encodes_running=len(encodes)))
                last_progress = time.time()
            time.sleep(0.5)
    except Abort as e:
        workers.stop()
        for proc in encodes.values():
            proc.terminate()
        progress(work, jobs, "stopped", t0, frames_total, dict(message=str(e)))
        sys.exit(str(e))
    finally:
        for s, h in old_handlers.items():
            signal.signal(s, h)

    # Join jobs, then add the source audio for the same range.
    listing = work / "segments.txt"
    listing.write_text("".join(f"file '{work / 'jobs' / f'j{i:03d}' / 'video.mkv'}'\n" for i in range(len(jobs))))
    video = work / "video.mkv"
    run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y", "-f", "concat", "-safe", "0",
         "-i", listing, "-c", "copy", video])
    result = work / "upscaled.mkv"
    whole = first == 0 and last == info["frames"]
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y", "-i", video]
    if info["audio"]:
        cut = [] if whole else ["-ss", f"{float(first / fps):.6f}", "-t", f"{float((last - first) / fps):.6f}"]
        # A partial range re-encodes the audio (a stream copy can't cut on our frame boundary).
        acodec = ["-c:a", "copy"] if whole else ["-c:a", "aac", "-b:a", "256k"]
        # No -shortest: it drops video frames when the audio ends a fraction early.
        cmd += [*cut, "-i", src, "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", *acodec]
    else:
        cmd += ["-c", "copy"]
    run([*cmd, result])
    total = media.frame_count(result)
    print(f"upscaled.mkv: {total} frames (expected {last - first})", flush=True)
    if total != last - first:
        sys.exit("frame count mismatch in upscaled.mkv")

    job_stats = []
    for i in range(len(jobs)):
        d = work / "jobs" / f"j{i:03d}"
        pieces = [json.loads(m.read_text()) for m in sorted((d / "pieces").glob("p*.json"))]
        done = json.loads((d / "done").read_text()) if (d / "done").exists() else {}
        job_stats.append(dict(job=f"j{i:03d}", frames=sum(len(plan.written_frames(p)) for p in jobs[i]),
                              worker=done.get("worker"), seconds=done.get("seconds"), pieces=pieces))
    timing = dict(frames=last - first, jobs=len(jobs), pieces=n_pieces, scene_cuts=len(cuts),
                  wall_seconds_this_run=round(time.time() - t0, 1), engine=a.engine, settings=settings,
                  gpus=a.gpus if eng["gpu"] else None, size=f"{out[0]}x{out[1]}", per_job=job_stats)
    write_json(work / "timing.json", timing)
    progress(work, jobs, "done", t0, frames_total)

    if a.compare == "yes" or (a.compare == "auto" and (last - first) / fps <= COMPARE_MAX_SECONDS):
        # Left: plain Lanczos of the source (same colour handling as the engines). Right: result.
        left = media.to_rgb_filter(info, out, a.deinterlace)
        run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
             "-ss", f"{media.seek_time(first, fps):.6f}", "-i", src, "-i", result, "-filter_complex",
             f"[0:v]{left},trim=end_frame={last - first},scale=out_color_matrix=bt709:out_range=tv,"
             f"format=yuv420p,setsar=1[l];"
             f"[1:v]scale=in_color_matrix=bt709:out_color_matrix=bt709,format=yuv420p,setsar=1[r];"
             f"[l][r]hstack=inputs=2[v]", "-map", "[v]",
             "-c:v", "libx264", "-crf", "14", "-preset", "medium", "-colorspace", "bt709",
             "-color_primaries", "bt709", "-color_trc", "bt709", work / "compare.mp4"])
    print(f"done: {work}", flush=True)


if __name__ == "__main__":
    main()
