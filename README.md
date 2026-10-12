# VideoUpscale

Restore and upscale old, compressed SD video with [SeedVR2](https://github.com/ByteDance-Seed/SeedVR) (7B, FP16), through numz's standalone CLI ([ComfyUI-SeedVR2_VideoUpscaler](https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler), pinned in `thed1/setup.sh`).

Code is edited on the PC and kept in Gitea. thed1 only gets a clean clone; the venv, upstream code and weights live in `~/VideoUpscale-deps`, and jobs write to `~/VideoUpscale-work`.

## Setup on thed1 (once)

```bash
git clone http://wilkinsnas.local:30008/sojun/VideoUpscale.git ~/VideoUpscale
sudo thed1-netopen 2h
~/VideoUpscale/thed1/setup.sh          # VAE + 7B fp16, ~17 GB
sudo thed1-netclose
```

`setup.sh` also applies `thed1/patches/*.patch` to the upstream checkout. Rerunning it is safe with the network closed; set `VIDEOUPSCALE_REINSTALL=1` to redo the pip installs.

**Patches:** `0001` raises the 7B VAE's conv/norm memory limits from 0.5 to 8 GiB. The 0.5 GiB default splits every layer into small pieces for small cards. With 64 GB cards, 8 GiB makes encode + decode about 13% faster: 4.17 + 9.21 s instead of 4.72 + 10.65 s per 21-frame 1440×1080 batch on a CMP 170HX. VAE peak goes from 26 to 39 GiB. `bench/vae_bench.py` measures this. cuDNN benchmark mode, the normal Conv3d path and channels-last didn't help (channels-last was 1.6× slower).

## Run

```bash
python3 ~/VideoUpscale/upscale.py SRC.mkv --work ~/VideoUpscale-work/test1 --start 830 --duration 20 --gpus 2
```

Output in the work dir:

- `upscaled.mkv`: x265 10-bit, BT.709 tagged, with the source audio
- `compare.mp4`: Lanczos on the left, SeedVR2 on the right
- `stills/`: three frames from the comparison
- `timing.json`, `seedvr2.log`

## What the pipeline does with SD

- **Interlacing:** `idet` checks the source; interlaced or telecined input is refused, since deinterlacing/IVTC isn't implemented yet.
- **Pixel aspect:** anamorphic SD (720×480 at SAR 8:9) is resampled to square pixels without dropping source pixels (720×540), so the model sees the right shapes.
- **Colour:** untagged SD is decoded as BT.601, and the output is converted and tagged BT.709. Upstream's own ffmpeg writer skips this and drops audio, so it writes PNGs and this script encodes.
- **Output size:** `--height` sets the short side; the default is 1080 (720×540 → 1440×1080).
