#!/usr/bin/env bash
# One-time setup on thed1: upstream SeedVR2 CLI at a pinned commit, a venv
# with the torch build known to work on the CMP 170HX driver, and the model
# weights, checked against Hugging Face's sha256.
#
# Everything lands in $VIDEOUPSCALE_DEPS (default ~/VideoUpscale-deps), never
# inside this repo checkout. Needs outbound network: run `sudo thed1-netopen 2h`
# first and `sudo thed1-netclose` after.
#
# Usage: thed1/setup.sh [model-file ...]   (default: VAE + 7B fp16)
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
DEPS="${VIDEOUPSCALE_DEPS:-$HOME/VideoUpscale-deps}"
UPSTREAM_URL=https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler.git
UPSTREAM_REV=4490bd1f482e026674543386bb2a4d176da245b9   # main, 2025-12-24 (v2.5.24)
TORCH_INDEX=https://download.pytorch.org/whl/cu130
HF_BASE=https://huggingface.co/numz/SeedVR2_comfyUI/resolve/main

log() { printf '[setup] %s\n' "$*"; }
mkdir -p "$DEPS/models/SEEDVR2"

# --- upstream code -----------------------------------------------------------
if [[ ! -d "$DEPS/seedvr2/.git" ]]; then
  log "cloning upstream SeedVR2"
  git clone -q "$UPSTREAM_URL" "$DEPS/seedvr2"
fi
if [[ "$(git -C "$DEPS/seedvr2" rev-parse HEAD)" != "$UPSTREAM_REV" ]]; then
  git -C "$DEPS/seedvr2" fetch -q origin
  git -C "$DEPS/seedvr2" -c advice.detachedHead=false checkout -q "$UPSTREAM_REV"
fi
log "upstream at $(git -C "$DEPS/seedvr2" rev-parse --short HEAD)"
# Our patches (thed1/patches/*.patch), reapplied from a clean tree every run.
git -C "$DEPS/seedvr2" checkout -q -- .
for p in "$HERE"/patches/*.patch; do
  git -C "$DEPS/seedvr2" apply "$p"
  log "applied $(basename "$p")"
done

# --- venv --------------------------------------------------------------------
# Installed once; rerun with VIDEOUPSCALE_REINSTALL=1 (and the network open) to redo.
if [[ ! -f "$DEPS/venv/.installed" || -n "${VIDEOUPSCALE_REINSTALL:-}" ]]; then
  if [[ ! -x "$DEPS/venv/bin/python" ]]; then
    log "creating venv"
    python3 -m venv "$DEPS/venv"
  fi
  PIP="$DEPS/venv/bin/pip"
  "$PIP" install -q --upgrade pip
  "$PIP" install -q torch==2.13.0 torchvision==0.28.0 --index-url "$TORCH_INDEX"
  # Headless OpenCV: the desktop build wants libGL, which this server doesn't need.
  grep -v '^opencv-python' "$DEPS/seedvr2/requirements.txt" > "$DEPS/requirements.thed1.txt"
  echo opencv-python-headless >> "$DEPS/requirements.thed1.txt"
  "$PIP" install -q -r "$DEPS/requirements.thed1.txt"
  touch "$DEPS/venv/.installed"
fi
"$DEPS/venv/bin/python" -c 'import torch; print("[setup] torch", torch.__version__, "cuda", torch.cuda.is_available(), torch.cuda.device_count(), "GPUs")'

# --- models ------------------------------------------------------------------
wanted=("$@")
[[ ${#wanted[@]} -eq 0 ]] && wanted=(ema_vae_fp16.safetensors seedvr2_ema_7b_fp16.safetensors)
for name in "${wanted[@]}"; do
  line="$(grep -E "^$name " "$HERE/models.txt" || true)"
  [[ -n "$line" ]] || { echo "unknown model $name (not in thed1/models.txt)" >&2; exit 1; }
  read -r _ size sha <<<"$line"
  dest="$DEPS/models/SEEDVR2/$name"
  if [[ -f "$dest" && "$(stat -c %s "$dest")" == "$size" ]]; then
    log "$name present"
    continue
  fi
  log "downloading $name ($((size / 1000000)) MB)"
  curl -fL --retry 5 -C - -o "$dest.part" "$HF_BASE/$name"
  [[ "$(stat -c %s "$dest.part")" == "$size" ]] || { echo "$name: size mismatch" >&2; exit 1; }
  echo "$sha  $dest.part" | sha256sum -c --quiet || { echo "$name: sha256 mismatch" >&2; exit 1; }
  mv "$dest.part" "$dest"
  log "$name verified"
done
log "done"
