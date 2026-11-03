"""Stand-in for numz's SeedVR2 inference_cli: bicubic upscale plus a constant
offset that alternates per call, so tests can see which piece made a frame and
check crossfades. Records every call in $FAKE_SEEDVR2_LOG."""
import argparse
import json
import os

import cv2
import numpy as np


class _Debug:
    enabled = False


debug = _Debug()
OFFSET = 0.05


def parse_arguments():
    p = argparse.ArgumentParser()
    p.add_argument("input")
    for flag in ("--model_dir", "--dit_model", "--color_correction", "--attention_mode",
                 "--dit_offload_device", "--vae_offload_device"):
        p.add_argument(flag)
    for flag in ("--resolution", "--batch_size", "--temporal_overlap", "--seed", "--prepend_frames"):
        p.add_argument(flag, type=int)
    for flag in ("--input_noise_scale", "--latent_noise_scale"):
        p.add_argument(flag, type=float)
    for flag in ("--cache_dit", "--cache_vae", "--compile_dit", "--compile_vae", "--debug"):
        p.add_argument(flag, action="store_true")
    return p.parse_args()


def _process_frames_core(frames_tensor, args, device_id, debug, runner_cache):
    assert device_id == "0" and args.cache_dit and args.dit_offload_device == "0"
    x = frames_tensor.a.astype(np.float32)
    if args.prepend_frames:  # upstream prepends mirrored frames and leaves them in the output
        x = np.concatenate([x[1:args.prepend_frames + 1][::-1], x])
    t, h, w, _ = x.shape
    if w >= h:
        size = (int(args.resolution * w / h) // 2 * 2, args.resolution)
    else:
        size = (args.resolution, int(args.resolution * h / w) // 2 * 2)
    calls = runner_cache.setdefault("calls", 0)
    runner_cache["calls"] = calls + 1
    off = OFFSET if calls % 2 else 0.0
    y = np.stack([cv2.resize(f, size, interpolation=cv2.INTER_CUBIC) for f in x])
    with open(os.environ["FAKE_SEEDVR2_LOG"], "a") as f:
        f.write(json.dumps(dict(frames=t, batch_size=args.batch_size, prepend=args.prepend_frames,
                                offset=off, pid=os.getpid())) + "\n")
    return np.clip(y + off, 0.0, 1.0)
