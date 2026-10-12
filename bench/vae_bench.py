#!/usr/bin/env python3
"""Time the SeedVR2 VAE alone (encode + decode of one batch) under different settings.

Run with the deps venv, on thed1:
  ~/VideoUpscale-deps/venv/bin/python bench/vae_bench.py --gpu 2 [--conv-mem inf] [--no-workaround] ...

The defaults reproduce what the upstream CLI does for the 7B config: bf16 compute,
causal slicing every 4 frames, 0.5 GiB conv/norm memory limits, direct cuDNN calls
with benchmark=False (its Conv3d memory-bug workaround).
"""
import argparse
import math
import os
import sys
import time
from pathlib import Path

DEPS = Path(os.environ.get("VIDEOUPSCALE_DEPS", Path.home() / "VideoUpscale-deps"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=2)
    ap.add_argument("--frames", type=int, default=21)
    ap.add_argument("--width", type=int, default=1440)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--conv-mem", default="0.5", help="GiB, or inf")
    ap.add_argument("--norm-mem", default="0.5", help="GiB, or inf")
    ap.add_argument("--split-size", default="4", help="causal slicing frames, or none")
    ap.add_argument("--no-workaround", action="store_true", help="use the normal Conv3d path")
    ap.add_argument("--cudnn-benchmark", action="store_true")
    ap.add_argument("--channels-last", action="store_true", help="channels_last_3d weights and input")
    ap.add_argument("--repeats", type=int, default=2)
    a = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)
    sys.path.insert(0, str(DEPS / "seedvr2"))
    import torch
    from omegaconf import OmegaConf
    from safetensors.torch import load_file
    from src.common.config import load_config, create_object
    import src.models.video_vae_v3.modules.causal_inflation_lib as cil

    if a.no_workaround:
        cil.NVIDIA_CONV3D_MEMORY_BUG_WORKAROUND = False
    torch.backends.cudnn.benchmark = a.cudnn_benchmark
    dtype = getattr(torch, a.dtype)
    dev = torch.device("cuda:0")

    root = DEPS / "seedvr2"
    main_cfg = load_config(str(root / "configs_7b/main.yaml"))
    vae_cfg = load_config(str(root / "src/models/video_vae_v3/s8_c16_t4_inflation_sd3.yaml"))
    model = create_object(OmegaConf.merge(main_cfg.vae.model, vae_cfg))
    state = load_file(str(DEPS / "models/SEEDVR2/ema_vae_fp16.safetensors"))
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"warning: {len(missing)} missing, {len(unexpected)} unexpected keys")
    model = model.to(dev, dtype).eval().requires_grad_(False)
    if a.channels_last:
        model = model.to(memory_format=torch.channels_last_3d)

    mem = lambda s: None if s == "inf" else float(s)
    model.set_memory_limit(conv_max_mem=mem(a.conv_mem), norm_max_mem=mem(a.norm_mem))
    if a.split_size != "none":
        model.set_causal_slicing(split_size=int(a.split_size), memory_device="same")
    else:
        model.use_slicing = False

    x = torch.rand(1, 3, a.frames, a.height, a.width, device=dev, dtype=dtype) * 2 - 1
    if a.channels_last:
        x = x.contiguous(memory_format=torch.channels_last_3d)

    def timed(fn):
        torch.cuda.synchronize()
        t = time.perf_counter()
        out = fn()
        torch.cuda.synchronize()
        return out, time.perf_counter() - t

    print(f"settings: {vars(a)} workaround_active={cil.NVIDIA_CONV3D_MEMORY_BUG_WORKAROUND}", flush=True)
    with torch.inference_mode():
        for r in range(a.repeats):
            torch.cuda.reset_peak_memory_stats()
            z, te = timed(lambda: model.encode(x).latent)
            peak_e = torch.cuda.max_memory_allocated() / 2**30
            torch.cuda.reset_peak_memory_stats()
            y, td = timed(lambda: model.decode(z).sample)
            peak_d = torch.cuda.max_memory_allocated() / 2**30
            print(f"run {r}: encode {te:6.2f} s (peak {peak_e:5.1f} GiB)  "
                  f"decode {td:6.2f} s (peak {peak_d:5.1f} GiB)  latent {tuple(z.shape)} out {tuple(y.shape)}",
                  flush=True)
            del z, y


if __name__ == "__main__":
    main()
