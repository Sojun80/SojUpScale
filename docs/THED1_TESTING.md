# Testing on thed1

Everything in this branch was verified on CPU only. The GPU path (the SeedVR2
runner calling the real model) has **not** been run yet. This is the order to
validate it, cheapest first. Nothing here touches mining configuration; stop
mining on the GPUs you use, as before.

## 0. Update the checkout

```bash
cd ~/VideoUpscale && git fetch && git checkout claude/vigilant-cannon-lx18vx
```

`thed1/setup.sh` is unchanged. Rerun it only to fetch the sharp weights
(network open):

```bash
~/VideoUpscale/thed1/setup.sh seedvr2_ema_7b_sharp_fp16.safetensors
```

## 1. CPU tests (2 minutes, no GPU)

```bash
cd ~/VideoUpscale && ~/VideoUpscale-deps/venv/bin/python -m unittest discover -s tests -t . -v
```

All should pass (the SeedVR2 runner test uses a fake model). A failure here
means a difference in ffmpeg/OpenCV versions; send the output.

## 2. Smoke test: one GPU, one short shot

```bash
python3 ~/VideoUpscale/upscale.py SRC.mkv --work ~/VideoUpscale-work/smoke \
    --start 830 --duration 3 --gpus 2
```

Check:
- `workers/gpu2.log` shows the model loading once and `piece 1/1 ... fps`.
- `upscaled.mkv` plays, has the expected frame count (printed), looks like the
  old output, and `compare.mp4` is aligned (no offset between the halves).
- `jobs/j000/pieces/p000.json` has `batch_size` and `gpu_peak_gib`.

If the model fails to load or runs out of memory, the error is in
`workers/gpu2.log`. The runner keeps the DiT and VAE resident on the GPU
between pieces (`--cache_dit --cache_vae` with the "offload" device set to the
same GPU); that is the one upstream code path we use that the old pipeline
didn't, so check it first.

## 3. Same range as the old measurement: speed

The old pipeline did 20 s of 720×480 → 1440×1080 in 260 s on four GPUs
(batch 45, 8 GiB VAE patch) = 2.3 fps. Repeat that range:

```bash
python3 ~/VideoUpscale/upscale.py SRC.mkv --work ~/VideoUpscale-work/speed \
    --start 830 --duration 20
```

`timing.json` → `wall_seconds_this_run`, and per piece `restore_s`. Expect the
4 workers to run different jobs at the same time. Note that a 20 s range is
only one to four jobs, so with few scene cuts some GPUs idle: for a fair
throughput number use a longer range (2–5 minutes), or compare per-GPU fps.

Expected differences (unmeasured):
- Faster per frame on long videos: the model loads once per GPU per run instead
  of once per GPU per 60 s segment.
- Slightly more work: ~3% padding frames from balanced batches, and pieces of
  long shots overlap by 12 frames.
- PNG writing is 16-bit (bigger files, slower writes), done in 16 threads while
  the GPU waits. If `write_s` per piece is large compared with `restore_s`,
  try `--png-depth 8` to see what it costs.
- RAM: each worker holds one piece. At the default 20 s pieces, about 20 GB per
  worker at 1080p (four workers ≈ 80 GB). Lower `--max-piece-seconds` if needed.

## 4. Quality comparison

Pick 5–8 short clips (5–10 s each) that show what matters: faces in close-up
and at a distance, hair, fabric, foliage or brickwork, fast motion and pans,
dark scenes, and heavy compression blocks. Include at least one long take
without cuts (to see piece and batch seams), and if you have any good HD
footage, one degraded copy (`--degrade dvd`) for a ground-truth check.

```bash
E=~/VideoUpscale-work/eval
python3 ~/VideoUpscale/eval/clips.py add SRC.mkv --eval $E --name face-close --start 812 --duration 8 --tags faces
python3 ~/VideoUpscale/eval/clips.py add SRC.mkv --eval $E --name pan-texture --start 1260 --duration 8 --tags motion,texture --crop 500,300,400,225
python3 ~/VideoUpscale/eval/clips.py add HD.mkv  --eval $E --name hd-dvd --start 60 --duration 6 --degrade dvd
python3 ~/VideoUpscale/eval/clips.py list --eval $E
```

`--crop x,y,w,h` (in output pixels) marks the region the report zooms into,
e.g. a face.

To include the previous pipeline as a candidate:

```bash
git -C ~/VideoUpscale worktree add ../SojUpScale-legacy d9715b5
```

Run the candidates. Each runs on its own GPU, four at a time; `legacy` takes
all four GPUs because its 4-GPU split is part of what is being compared:

```bash
python3 ~/VideoUpscale/eval/run.py --eval $E --only lanczos,7b,7b-sharp,7b-wavelet,legacy
python3 ~/VideoUpscale/eval/run.py --eval $E --only 7b-ov9,7b-b81-ov9,7b-prepend4,7b-unbalanced,7b-nocolor,7b-noise
~/VideoUpscale-deps/venv/bin/python ~/VideoUpscale/eval/report.py --eval $E --flow
```

Copy `$E/report/` to the PC and open `index.html`. The report has, per clip:
a labelled side-by-side `grid.mp4` (first tile: Lanczos of the source),
`crops.mp4` for the region of interest, still sheets at 2× pixel zoom, an x-t
slice (one row over time: flicker and seams show as horizontal breaks), and a
table of runtime, GPU memory, flicker beyond the source's, seam ratio,
fidelity to the source and colour shift. Re-running `run.py` skips runs whose
definition hasn't changed.

What to decide from it, in this order:
1. Does `7b` look clearly better than `lanczos` on faces and textures, without
   invented detail (compare with the source tile, and with the reference on
   degraded clips)? If not, settings won't save it, see the model notes in
   `docs/REPORT.md`.
2. `7b` vs `legacy`: same model and settings; differences come from the
   pipeline (no mid-shot GPU splits, balanced batches, 16-bit, exact colour
   path). Look at long takes around the frames marked as seams.
3. `7b` vs `7b-sharp`: detail vs artefacts (ringing, crunchy textures, faces).
4. Colour: `lab` vs `wavelet` vs `none`; look for colour pumping in the x-t
   slice and the flicker+ curve.
5. Seams: `7b-ov9` and `7b-b81-ov9` vs `7b`, seam ratio and long takes; and
   whether the extra compute is worth it.

Then set the winners as defaults in `upscale.py` (a one-line change per
setting) and note the evidence in `docs/REPORT.md`.

## 5. Full-length run

```bash
python3 ~/VideoUpscale/upscale.py SRC.mkv --work ~/VideoUpscale-work/full
```

Interrupt it (Ctrl-C) once mid-run and start it again: finished pieces must not
be redone (`piece N/M` lines resume where they stopped), and `upscaled.mkv`
must have the source's frame count. `progress.json` in the work directory shows
progress; creating a file named `cancel` there stops the run cleanly.
