#!/usr/bin/env python3
"""SeedVR2 engine worker. Runs in the deps venv on one GPU (the orchestrator
sets CUDA_VISIBLE_DEVICES), using numz's SeedVR2 code from $VIDEOUPSCALE_DEPS/seedvr2.

Instead of the upstream CLI's file loop, this calls its per-clip function
(inference_cli._process_frames_core) once per piece, with the model cached on
the GPU between pieces and jobs. That gives us, per piece:
  * a batch size balanced to the piece length (no short last batch),
  * frames read by our own ffmpeg chain (exact range, explicit colour matrix,
    16-bit RGB instead of OpenCV's 8-bit decode),
  * rounded 16-bit PNG output (upstream truncates to 8 bits).
All model settings are passed through unchanged; see ENGINE_VERSION below.
"""
import os
import sys
import time
from argparse import Namespace
from pathlib import Path

import common  # noqa: F401  (puts the repo on sys.path)
import framework
from common import log, run_worker

from sojupscale import plan

# Bump when a change here alters output pixels: it is part of the resume identity.
ENGINE_VERSION = 1

DEPS = Path(os.environ.get("VIDEOUPSCALE_DEPS", Path.home() / "VideoUpscale-deps"))


def upstream_args(cli, s, out_short_side):
    """Upstream CLI defaults, plus our settings, from its own argument parser."""
    argv = ["inference_cli.py", "unused-input",
            "--model_dir", str(DEPS / "models/SEEDVR2"), "--dit_model", s["model"],
            "--resolution", str(out_short_side), "--batch_size", str(s["batch_size"]),
            "--temporal_overlap", str(s["overlap"]), "--color_correction", s["color_correction"],
            "--attention_mode", s["attention"], "--seed", str(s["seed"]),
            "--prepend_frames", str(s["prepend_frames"]),
            "--input_noise_scale", str(s["input_noise"]), "--latent_noise_scale", str(s["latent_noise"]),
            # Keep both models resident on this GPU between pieces: cache them, and
            # "offload" to the GPU they are already on (a no-op move upstream).
            "--cache_dit", "--cache_vae", "--dit_offload_device", "0", "--vae_offload_device", "0"]
    if s.get("compile"):
        argv += ["--compile_dit", "--compile_vae"]
    if s.get("debug"):
        argv.append("--debug")
    saved = sys.argv
    sys.argv = argv
    try:
        return cli.parse_arguments()
    finally:
        sys.argv = saved


def setup(job):
    s = job["engine"]["settings"]
    # Upstream reads --cuda_device from sys.argv at import time; give it none.
    sys.argv = sys.argv[:1]
    sys.path.insert(0, str(DEPS / "seedvr2"))
    import torch
    import inference_cli as cli
    out_w, out_h = job["output"]["size"]
    args = upstream_args(cli, s, min(out_w, out_h))
    cli.debug.enabled = bool(s.get("debug"))
    model_path = DEPS / "models/SEEDVR2" / s["model"]
    if not model_path.exists():
        raise FileNotFoundError(f"{model_path} missing; run thed1/setup.sh {s['model']}")
    log(f"SeedVR2 {s['model']} on {torch.cuda.get_device_name(0)}; batch <= {s['batch_size']}, "
        f"overlap {s['overlap']}, colour {s['color_correction']}, balanced={s['balanced_batches']}")
    return dict(cli=cli, torch=torch, args=args, cache={}, settings=s)


def make_restore(state):
    cli, torch, s = state["cli"], state["torch"], state["settings"]

    def restore(x, piece):
        n, pre = len(x), s["prepend_frames"]
        a = Namespace(**vars(state["args"]))
        if s["balanced_batches"]:
            a.batch_size = plan.balanced_batch(n + pre, s["batch_size"], s["overlap"])
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        out = cli._process_frames_core(torch.from_numpy(x).to(torch.float16), a, "0", cli.debug, state["cache"])
        if pre:
            out = out[pre:]  # upstream leaves prepended frames in for its callers to drop
        restore.last_stats = dict(batch_size=a.batch_size, gpu_peak_gib=round(torch.cuda.max_memory_allocated() / 2**30, 2),
                                  gpu_seconds=round(time.time() - t0, 2))
        return out

    restore.last_stats = {}
    return restore


def process_job(job, state):
    if "restore" not in state:
        state["restore"] = make_restore(state)
    framework.process_job(job, state["restore"])


if __name__ == "__main__":
    run_worker(process_job, setup=setup)
