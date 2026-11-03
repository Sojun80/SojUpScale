# SojUpScale: restoration quality report

Status as of this branch (`claude/vigilant-cannon-lx18vx`). The work was done
in a cloud container **without a GPU**. Everything marked *proven* was shown
by reading the code or by CPU tests in this repo. Nothing here claims a
visual improvement or a GPU speed: those need the experiments in
[`THED1_TESTING.md`](THED1_TESTING.md).

## 1. Summary

- **Model:** SeedVR2-7B is still the strongest open model for restoring
  degraded real-world video, and it is Apache-2.0. Keep it. Test the sharp
  weights. FlashVSR v1.1 is the only alternative worth integrating later, as a
  faster, steadier, less inventive tier, not as a quality upgrade. See section 3.
- **Pipeline:** the old pipeline had structural problems that hurt temporal
  quality regardless of model settings. Every 60 s segment was cut into four
  independent runs mid-shot. 45-frame batches crossed scene cuts. Each run
  ended with a short last batch. There were also two frame-accuracy bugs, an
  8-bit truncation, and a colour path that was only correct for SD sources.
  All of these are fixed (section 4).
- **Settings:** the model settings most likely to matter (sharp vs standard,
  colour correction, batch overlap/length, prepend frames) are left at their old
  values. There is no visual evidence yet to change them. They are lined up as
  one-variable candidates in `eval/candidates.json`.
- **Evaluation:** `eval/` makes exact clips (optionally degraded copies of good
  footage, for ground truth), runs candidates in parallel on the GPUs while
  recording settings, runtime and GPU memory, and builds an HTML report. The
  report has synchronised labelled grids, zoomed crops, still sheets, x-t slices,
  and flicker, seam and fidelity measures.

## 2. Assessment of the old pipeline

The old pipeline was `upscale.py` at `d9715b5`, calling numz's SeedVR2 CLI at
`4490bd1`. The upstream code paths referred to below are in that CLI.

### Proven problems

| # | Problem | Evidence | Effect | Status |
|---|---------|----------|--------|--------|
| 1 | **Mid-shot GPU splits.** With `--cuda_device 0,1,2,3` upstream gives each GPU a contiguous quarter of the clip (`_gpu_processing`). The four runs are independent and joined with a `--temporal_overlap 3` crossfade. With 60 s segments that is a seam every ~15 s, wherever it falls. | code | Each run starts cold (the causal VAE's first frame, no temporal context). Detail and colour can jump at the join. | **Fixed.** Shots are never split across GPUs. Only shots longer than 20 s are split, with 12-frame crossfades. |
| 2 | **Batches crossed scene cuts.** A 60 s segment holds many shots, and 45-frame batches ran straight through cuts. | code | Temporal attention mixes two unrelated shots. LAB colour matching (row 5) computes one histogram over both shots. | **Fixed.** One model call per shot (or piece of a shot). |
| 3 | **Short last batch.** Upstream steps through a clip in `batch - overlap` frames, so the last batch is whatever is left. At batch 45 / overlap 3 it can be 1–5 frames (worst case over lengths 46–1000, `tests/test_plan.py`). | test | The last frames of every run get little temporal context. | **Fixed.** Balanced batches give the same number of batches at near-equal length. The worst last batch over 46–1000 frames is ≥ 20 frames, for ~2.8% extra (padding) frames. `--no-balanced-batches` restores the old behaviour. |
| 4 | **Batch seams are almost hard cuts.** Upstream's overlap blend is a Hann crossfade over the middle third of the overlap (`blend_overlapping_frames`). With overlap 3 that is a single 50/50 frame, every 42 frames. | code | Possible periodic "pumping" every ~1.4 s. | **Not changed.** Needs GPU evidence; candidates `7b-ov9` and `7b-b81-ov9`. |
| 5 | **LAB colour correction is per batch.** `lab_color_transfer` first applies the wavelet fix, then rank-matches a\*/b\* (and 20% of L\*) histograms over *all pixels of all frames in the batch*. | code | The colour mapping changes from batch to batch, a plausible source of colour/brightness pumping, worse across cuts (row 2). | **Not changed.** Candidates `7b-wavelet` (per-pixel low-frequency transfer, no batch statistics) and `7b-nocolor`. |
| 6 | **8-bit truncation.** Upstream writes `(x * 255).astype(uint8)`, which truncates. Output is darker by half a step on average, quantised to 8 bits, then encoded to 10-bit x265. The model computes in bf16, which carries ~9+ bits. | code, test | Small. Possible banding in dark gradients. | **Fixed.** Rounded 16-bit PNGs (`--png-depth 8` to compare). |
| 7 | **Scene cuts one frame late** for ranges not starting at frame 0. Timestamps after the half-frame-early seek were rounded with Python's `round`, so 45.5 → 46. | experiment (`cuts.mp4`: cut at 60 reported as 61) | Joins put the new shot's first frame in the old segment. | **Fixed.** Uses ffmpeg's frame counter. |
| 8 | **`-shortest` dropped video frames** when the audio ended a fraction earlier than the video. | test (4 frames lost on a synthetic clip) | Missing frames at the end. | **Fixed.** The frame count is now checked and a mismatch is an error. |
| 9 | **Colour matrix assumed SD.** Prep re-tagged every source as SMPTE 170M and left the YUV→RGB conversion to OpenCV's decoder (BT.601, 8-bit). | code | Correct for SD. A BT.709 (HD) source would get a colour shift. | **Fixed.** The source's tag is used (or BT.601 for untagged SD, BT.709 above 576 lines), applied explicitly by ffmpeg, in 16-bit. |
| 10 | **Model reloaded for every segment** on all four GPUs. | code | Speed only (the README's numbers suggest ~15% overhead on a 20 s range). | **Fixed.** One persistent worker per GPU. |
| 11 | **Resume could reuse incompatible results.** Only the frame range was checked. | code | Changing model, settings or even the source file reused old segments silently. | **Fixed.** `manifest.json` (below). |

### Checked and ruled out

- **Frame-accurate seeking in upstream's GPU split.** Each GPU worker seeks
  with OpenCV `CAP_PROP_POS_FRAMES`. On a lossless x264 file like the old
  `prep.mp4`, and on a long-GOP x264 file with B-frames, every tested seek
  (15 positions each) returned exactly the frame a sequential decode gives
  (OpenCV 5.0, ffmpeg 6.1). No misalignment at the GPU joins.
- **Pixel aspect.** 720×480 SAR 8:9 → 720×540, never dropping source pixels.
  Correct, kept.

### Open questions (need GPU runs)

- Standard vs sharp 7B weights, on faces especially.
- Colour correction: `lab` vs `wavelet` vs `none` (row 5).
- Overlap and batch length (row 4), and the GPU memory for batch 81.
- `--prepend-frames` for shot starts, and light `--input-noise` on heavily
  compressed sources.
- Whether rows 3 and 6 are *visible*. They are correct fixes, but may be small.
- Upstream upsamples the model input with torchvision bicubic (antialiased)
  inside the model call. Pre-scaling with Lanczos instead was considered and
  **not** done: the model was trained on its own resize, and Lanczos ringing
  could be read as detail.
- Deinterlacing (`--deinterlace bwdif|fieldmatch`) is implemented as ffmpeg
  filters but not tried on real interlaced material.

## 3. Generative restoration models

Speed claims below come from each paper's own comparisons. Quality rankings
come from published tables and secondary comparisons, **not** from our
footage. Nothing has been run on thed1.

| Model | What it is | Licence | Fit for thed1 | Verdict |
|-------|------------|---------|---------------|---------|
| **SeedVR2-7B** (ByteDance, ICLR 2026) | One-step diffusion transformer for restoration (diffusion adversarial post-training). 7B, fp16 ~16.5 GB. | Apache-2.0 (code and weights, per the HF model cards) | Running now. 64 GB cards fit batch 45 at 1080p with room to spare. | **Keep as the engine.** Strongest open option for degraded real-world footage in the comparisons found. It reconstructs detail rather than sharpening. Risks: invented texture or face detail when the source is very poor, and some reports of stripe artefacts with the 7B. The HF card calls the checkpoints "prototype" models. |
| SeedVR2-7B **sharp** | Same model, weights tuned towards detail. | Apache-2.0 | Drop-in (`--model`). | **Test.** No source compares it with standard 7B for flicker or faces. Candidate `7b-sharp`. |
| SeedVR2-3B | Smaller sibling. | Apache-2.0 | Drop-in, about 2× faster. | Candidate for a "fast preview" tier later. Not a quality upgrade. |
| SeedVR (CVPR 2025) | The multi-step predecessor. | Apache-2.0 | Far slower. | Superseded by SeedVR2. |
| **FlashVSR v1.1** (2025) | One-step streaming VSR on Wan2.1-1.3B with block-sparse attention. The authors report ~17 fps at 768×1408 on one A100, 11.8× faster than SeedVR2-3B. | Apache-2.0 (code and weights) | Needs its own venv. Full speed needs the Block-Sparse-Attention kernels compiled for sm_80; some ports fall back to dense attention and differ from the official output. Built for **4×**, while our SD→1080p is 2×. Training data (VSR-120K) is unreleased. | **Best alternative, for a different tier.** Secondary comparisons describe it as more conservative and temporally steadier than SeedVR2, which recovers more fine detail. A good "fast/stable" option. Unlikely to beat 7B on badly degraded SD, but that is untested. Integrate only if the SeedVR2 evaluation shows flicker that settings can't fix. |
| FlashVSR+ (arXiv 2610.17552) | Follow-up: lower latency and memory than FlashVSR. | unverified | unverified | Watch. Weights not confirmed available. |
| Vivid-VR (Alibaba) | CogVideoX1.5-5B-based restoration with ControlNet-style conditioning. | Code Apache-2.0, **base model under the CogVideoX licence** | Multi-step sampling: slow at 1080p. | Not now (licence terms and speed). |
| STAR | Text-to-video-prior VSR. I2VGen-XL variant (MIT) and CogVideoX-5B variant (CogVideoX licence). | mixed | Multi-step (15 steps in FlashVSR's comparison). | Not now. Slower, and behind SeedVR2 in published comparisons. |
| Upscale-A-Video | SD-x4-based video upscaler, 30 steps. | S-Lab (non-commercial) | Slow. | No. |
| DOVE, InfVSR, DUO-VSR, OSDEnhancer | 2025–26 research VSR (one-step or streaming). DOVE scores higher MUSIQ but lower PSNR than SeedVR2-3B in the FlashVSR+ table. | unverified | unverified | Watch. Not mature enough to bet on. |
| Real-ESRGAN, BasicVSR++ etc. | Non-generative or GAN SR. | various | easy | Not generative reconstruction. The `lanczos` engine is the conventional baseline. |
| Topaz Starlight | The target look. | proprietary | – | Not usable. Exports can still be compared as `"file"` candidates in `eval/run.py`. |

About **thed1's GPUs:** the CMP 170HX is a GA100 (compute 8.0), so bf16, SDPA
and flash-attention work. Its host link is reportedly PCIe 1.1 ×4. That makes
moving weights over PCIe expensive, so the new workers keep the model
resident instead of offloading it between phases (`--cache_dit/--cache_vae`
with the offload device set to the same GPU).

Sources: [SeedVR repo](https://github.com/ByteDance-Seed/SeedVR),
[SeedVR2-7B card](https://huggingface.co/ByteDance-Seed/SeedVR2-7B),
[FlashVSR repo](https://github.com/OpenImagingLab/FlashVSR),
[FlashVSR paper](https://arxiv.org/pdf/2510.12747),
[FlashVSR+ paper](https://arxiv.org/html/2610.17552v1),
[Vivid-VR card](https://huggingface.co/csbhr/Vivid-VR/blob/main/README.md),
[STAR card](https://huggingface.co/SherryX/STAR/blob/64cb4fe18b403cdcdf291f32d6f95beb0d0a507b/README.md),
[InfVSR](https://arxiv.org/pdf/2510.00948),
[DUO-VSR](https://arxiv.org/pdf/2603.22271),
[OSDEnhancer](https://arxiv.org/html/2601.20308),
[SeedVR2 vs FlashVSR](https://upsampler.com/blog/seedvr-vs-flashvsr-ai-video-super-resolution-2026),
[RunComfy SeedVR2 notes](https://www.runcomfy.com/models/community/seedvr2).

## 4. What changed

```
upscale.py              orchestrator (stdlib): probe, plan, queue, workers, encodes, join
sojupscale/plan.py      shots -> pieces -> jobs; balanced batch sizes; plan checker
sojupscale/media.py     ffprobe/ffmpeg helpers (colour, SAR, exact reads, scene cuts)
sojupscale/manifest.py  resume identity
engines/common.py       job-queue protocol for workers
engines/framework.py    read piece -> restore -> crossfade -> 16-bit PNGs
engines/seedvr2_runner.py  SeedVR2 via upstream's per-clip function, model cached
engines/lanczos_runner.py  CPU reference engine
eval/clips.py, run.py, report.py, candidates.json
tests/                  unittest, CPU only
```

- **Planning.** Scene cuts (ffmpeg `scdet`, same threshold as before) split
  the range into shots. Shots longer than `--max-piece-seconds` (20 s)
  become pieces overlapping by `--piece-overlap` (12) frames. Inside a job they
  are crossfaded with upstream's own curve; between jobs they are cut mid-overlap,
  with context on both sides. Shots are grouped into jobs of about
  `--segment-seconds` (20 s). `plan.check` verifies that every frame is written
  exactly once and that no piece crosses a cut, before anything runs.
- **Workers.** One process per GPU loads SeedVR2 once and claims jobs from
  `WORK/queue` by atomic rename, so faster GPUs simply take more jobs. Each
  piece is one call of upstream's `_process_frames_core` with a balanced batch
  size. Frames are read by our ffmpeg chain, results are written as rounded
  16-bit PNGs in 16 threads, and a marker file is left per piece.
- **Encodes** run on the CPU while the GPUs continue (`--encoders`), checked
  frame-for-frame, then their PNGs are deleted. Output: `upscaled.mkv`
  (x265 10-bit BT.709, source audio), `compare.mp4` for short ranges, as before.
- **Resume safety.** `WORK/manifest.json` holds the source's content identity
  (size + SHA-256 of three 4 MiB samples), range, sizes, colour handling,
  deinterlacing, plan parameters, engine settings, model SHA-256, runner
  version, upstream git revision, patch hash and encode settings. A rerun with
  any difference is refused and the differing keys are listed (`--restart`
  discards). Work directories from the old version are refused too. Resume is
  per piece within a job.
- **Replaceable engine.** An engine is a worker script that follows
  `engines/common.py`'s queue protocol, plus one line in `ENGINES` in
  `upscale.py`. Model engines get reading, crossfades and PNG writing from
  `engines/framework.py` and only supply `restore(frames) -> frames`. The
  `lanczos` engine is the conventional baseline and lets the whole pipeline
  run in tests without a GPU.
- **For the Windows app later.** `progress.json` (state, frames done,
  fraction, elapsed) is updated every few seconds. A `cancel` file in the work
  directory, or SIGINT/SIGTERM, stops cleanly and the run resumes later. The
  eval report already covers the "compare outputs" step.
- **Kept.** Four-GPU operation, scene-aware joins, audio handling, SAR
  handling, BT.709 10-bit output, background encoding, interlace refusal, both
  upstream patches, `thed1/setup.sh` and the CLI flags.

### Verification done here (CPU)

`python3 -m unittest discover -s tests -t .` covers:
- **Planner invariants.** 300 random plans, plus long takes and cuts at
  boundaries.
- **End-to-end with the lanczos engine.** Anamorphic SD with audio and two
  cuts. Every output frame is compared with a direct Lanczos render of the same
  source frame (min PSNR > 38 dB at x265 CRF 4: no frame shifts at any join).
  Also: resume with nothing redone, refusal of a changed setting and a replaced
  source, `--restart`, a partial range starting one frame after a cut, and an
  interrupted job resuming at its missing piece.
- **SeedVR2 runner against a fake upstream.** Batch sizes, prepend trimming,
  16-bit output, and the crossfade weights measured on the written frames. A
  mutation check confirmed the test fails when the crossfade is broken.

The tests found and fixed three bugs during development: duplicated frames
after `trim`, the `-shortest` frame loss, and the scene-cut off-by-one.

## 5. Evaluation workflow

See [`THED1_TESTING.md`](THED1_TESTING.md#4-quality-comparison) for the
commands. In short:

1. `eval/clips.py add` stores the exact frame range losslessly (FFV1), with
   tags and an optional crop of interest. With `--degrade dvd|web360|vhs` it
   degrades good footage and keeps the original as a reference.
2. `eval/run.py` runs every candidate on every clip, one GPU each, in
   parallel. It records the command, candidate definition, git and upstream
   revisions, model hash, wall time, fps, GPU name and peak memory. A candidate
   can also be another command (e.g. the old pipeline in a git worktree) or a
   finished file from elsewhere.
3. `eval/report.py` builds the HTML report. Per clip: labelled synchronised
   grid video, crop video, still sheets at 2× pixel zoom, an x-t slice, and a
   per-frame flicker chart with each run's seams marked. The table holds
   flicker beyond the source's own, seam jump, fidelity to the source, colour
   shift, ref PSNR/SSIM for degraded clips, and optional optical-flow warp
   error.

The report states plainly that its numbers only point at where to look. The
decision comes from watching the grid videos and stills. On synthetic CPU
runs, the measures separate as intended: a fake engine with a brightness step
at every piece join has flicker+ p99 ≈ 9 and seam jump 5.8–7.5 (8-bit steps),
while Lanczos and the same fake engine without joins sit at ≈ 0 (seam jump
0.01–0.03), and the seams show as horizontal bands in the x-t slice. Those runs check that the tooling
works and say nothing about model quality.

## 6. Proven, untested, next

**Proven (CPU):** the pipeline restructure and its frame accounting, resume
and refusal behaviour, the fixed bugs (rows 7, 8, 9 and the trim duplication),
balanced batches, 16-bit rounding, and the evaluation tooling end to end.

**Untested (needs thed1):**
- The SeedVR2 runner with the real model. In particular, keeping both models
  resident via upstream's cache path, and `_process_frames_core` called once
  per piece.
- Speed and GPU memory of the new layout.
- Every visual claim: whether the structural fixes are visible, and which
  settings look best.
- Deinterlacing on real interlaced material.

**Next, in order:**
1. Run `THED1_TESTING.md` steps 1–3 (tests, smoke, speed).
2. Build 5–8 clips (faces, hair and fabric, foliage and brick, fast motion,
   dark scenes, heavy blocking, one long take, one degraded HD) and run
   `lanczos, 7b, 7b-sharp, 7b-wavelet, legacy` first.
3. Pick defaults from the evidence (colour correction, overlap/batch, sharp or
   not) and record the decision here.
4. Only if seams or flicker remain: try FlashVSR v1.1 as a second engine
   (`engines/flashvsr_runner.py`, its own venv), compared in the same report.
5. Then the Windows front end: open → quality preset (= a candidate's settings)
   → preview (a 5 s range with `--start/--duration`) → process (thed1 over
   SSH, reading `progress.json`, cancelling with the `cancel` file) → save.
