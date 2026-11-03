# VideoUpscale

Restore and upscale old, compressed SD video with [SeedVR2](https://github.com/ByteDance-Seed/SeedVR) (7B, FP16), using numz's SeedVR2 code ([ComfyUI-SeedVR2_VideoUpscaler](https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler), pinned in `thed1/setup.sh`).

Code is edited on the PC and kept in Gitea. thed1 only gets a clean clone; the venv, upstream code and weights live in `~/VideoUpscale-deps`, and jobs write to `~/VideoUpscale-work`.

- [`docs/REPORT.md`](docs/REPORT.md): what limits quality, the model landscape, what changed, and what is proven vs untested.
- [`docs/THED1_TESTING.md`](docs/THED1_TESTING.md): how to validate on thed1 and compare candidates.

## Setup on thed1 (once)

```bash
git clone http://wilkinsnas.local:30008/sojun/VideoUpscale.git ~/VideoUpscale
sudo thed1-netopen 2h
~/VideoUpscale/thed1/setup.sh          # VAE + 7B fp16, ~17 GB
sudo thed1-netclose
```

`setup.sh` also applies `thed1/patches/*.patch` to the upstream checkout. Rerunning it is safe with the network closed; set `VIDEOUPSCALE_REINSTALL=1` to redo the pip installs. To fetch other weights, name them: `setup.sh seedvr2_ema_7b_sharp_fp16.safetensors`.

**Patches:** `0001` raises the 7B VAE's conv/norm memory limits from 0.5 to 8 GiB. The 0.5 GiB default splits every layer into small pieces for small cards. With 64 GB cards, 8 GiB makes encode + decode about 13% faster: 4.17 + 9.21 s instead of 4.72 + 10.65 s per 21-frame 1440×1080 batch on a CMP 170HX. VAE peak goes from 26 to 39 GiB. `bench/vae_bench.py` measures this. cuDNN benchmark mode, the normal Conv3d path and channels-last didn't help (channels-last was 1.6× slower). `0002` speeds up upstream's own PNG writer, which only the upstream CLI uses now.

## Run

```bash
python3 ~/VideoUpscale/upscale.py SRC.mkv --work ~/VideoUpscale-work/full           # whole file, all 4 GPUs
python3 ~/VideoUpscale/upscale.py SRC.mkv --work ~/VideoUpscale-work/t --start 830 --duration 20
```

**How the work is split.**
- **Shots:** scene cuts split the range into shots, and the model never sees across a cut.
- **Pieces:** a shot longer than `--max-piece-seconds` (20) becomes overlapping pieces that are crossfaded.
- **Jobs:** shots are grouped into jobs of about `--segment-seconds` (20).
- **Workers:** each GPU runs one worker that loads the model once and takes jobs from a queue. Within a piece, SeedVR2's batches (`--batch-size` 45, `--overlap` 3) are balanced to the piece length, so there is no short last batch.
- **Encoding:** a finished job is encoded to x265 on the CPU while the GPUs carry on, and its 16-bit PNGs are deleted afterwards.

**Resume** by rerunning the same command: finished pieces are skipped. `manifest.json` records the source's content and every setting that affects the pixels. A rerun with a different source, model or setting is refused with the differences listed (`--restart` discards the old results). `progress.json` shows progress. Creating a file named `cancel` in the work directory, or pressing Ctrl-C, stops the run cleanly.

Output in the work dir:

- `upscaled.mkv`: x265 10-bit, BT.709 tagged, with the source audio (stream-copied for a whole file)
- `compare.mp4`: Lanczos on the left, SeedVR2 on the right; made automatically for ranges up to 120 s
- `plan.json`: scene cuts, pieces and jobs
- `timing.json` (per piece: batch size, GPU seconds, GPU peak memory), `workers/gpuN.log`

**RAM:** each worker holds one piece, about 20 GB at 1080p for a 20 s piece. Lower `--max-piece-seconds` if the four workers don't fit.

**Engines:** `--engine seedvr2` (default) or `--engine lanczos` (plain resize on the CPU, the comparison baseline). An engine is a worker script in `engines/`; see `engines/common.py`.

**Measured on thed1 (old pipeline, before this restructure):** 20 s of 720×480 → 1440×1080 with 7B FP16 took 841 s on one GPU (batch 21) and 260 s on four (batch 45, 8 GiB VAE patch). That's 2.3 fps. The new layout has not been timed yet.

## What the pipeline does with SD

- **Interlacing:** `idet` checks the source. Interlaced or telecined input is refused unless `--deinterlace bwdif` (video) or `--deinterlace fieldmatch` (telecined film, no decimation) is given, or `--allow-interlaced`.
- **Pixel aspect:** anamorphic SD (720×480 at SAR 8:9) is resampled to square pixels without dropping source pixels (720×540), so the model sees the right shapes.
- **Colour:** the source's matrix tag is used. Untagged SD is treated as BT.601 and untagged larger video as BT.709. ffmpeg converts to 16-bit RGB for the model. The output is converted and tagged BT.709.
- **Output size:** `--height` sets the short side; the default is 1080 (720×540 → 1440×1080).

## Comparing restoration candidates

```bash
python3 eval/clips.py add SRC.mkv --eval ~/VideoUpscale-work/eval --name faces1 --start 812 --duration 8 --tags faces
python3 eval/run.py --eval ~/VideoUpscale-work/eval --only lanczos,7b,7b-sharp
~/VideoUpscale-deps/venv/bin/python eval/report.py --eval ~/VideoUpscale-work/eval
```

See `docs/THED1_TESTING.md` for the full procedure and `eval/candidates.json` for the candidates.

## Tests

```bash
python3 -m unittest discover -s tests -t .      # CPU only; needs ffmpeg (+ numpy/opencv for the runner tests)
```
