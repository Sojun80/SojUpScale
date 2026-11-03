#!/usr/bin/env python3
"""Compare the candidates' outputs for each clip: videos to watch, stills and
temporal slices to look at, and numbers that point at where to look.

  python eval/report.py --eval DIR [--clips a,b] [--flow] [--stills 0.2,0.5,0.8]

Writes DIR/report/index.html plus, per clip, DIR/report/CLIP/:
  grid.mp4     all candidates side by side, frame-synchronised, labelled; the
               first tile is plain Lanczos of the source
  crops.mp4    the clip's region of interest (clip.json "crop"), 2x pixel zoom
  stillN.png   the same frame from every candidate, region of interest zoomed
  xt.png       x-t slice: one image row over time, per candidate. Steady
               detail gives straight vertical streaks; flicker, crawling
               texture and batch seams show as horizontal breaks.

Metrics (per candidate, against the same clip):
  flicker+     temporal change beyond what plain Lanczos of the source has, on
               an 8x-downscaled luma (brightness/colour pumping) and at full
               resolution (texture shimmer). Lower is steadier; 0 = as steady
               as the source itself.
  seam jump    peak flicker+ around the run's own batch/piece/job boundaries,
               minus the median flicker+ elsewhere (scene cuts excluded), in
               8-bit steps. ~0 = seams invisible to this measure; clearly
               positive = look at the frames marked in the chart.
  fidelity     PSNR of the output shrunk back to the source size against the
               source: how much the restoration departs from what was there.
               Very low = invented structure or shifted colour; it is not a
               quality score (sharper restorations legitimately lose a bit).
  colour shift mean RGB difference to the source, in 8-bit steps.
  ref PSNR/SSIM  only for degraded clips: against the original before degradation.
  warp err+    (--flow) error after motion-compensating the previous output frame
               with optical flow of the source, beyond Lanczos's.
None of these measure perceived quality: watch grid.mp4 and the stills.

Needs numpy and opencv (both are in the SeedVR2 venv).
"""
import argparse
import html
import json
import math
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from sojupscale import media, plan  # noqa: E402

LUMA = np.array([0.2126, 0.7152, 0.0722], np.float32)
COLOURS = ["#888888", "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#42d4f4", "#f032e6",
           "#9a6324", "#469990", "#800000", "#000075"]


def ffmpeg(*args, quiet=True):
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y", *map(str, args)]
    return subprocess.run(cmd, capture_output=quiet)


def frames(path, vf, size):
    """Yield float32 RGB frames in [0, 1] of `path` after filter `vf`."""
    w, h = size
    proc = subprocess.Popen(["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-i", str(path),
                             "-map", "0:v:0", "-vf", vf + ",format=rgb48le", "-f", "rawvideo", "-"],
                            stdout=subprocess.PIPE)
    n = w * h * 6
    try:
        while True:
            buf = proc.stdout.read(n)
            if len(buf) < n:
                break
            yield np.frombuffer(buf, np.uint16).reshape(h, w, 3).astype(np.float32) / 65535.0
    finally:
        proc.stdout.close()
        proc.wait()


def out_filter(size, flags="lanczos"):
    """Our outputs are BT.709 limited range (any engine)."""
    return (f"scale={size[0]}:{size[1]}:flags={flags}+accurate_rnd:in_color_matrix=bt709:in_range=tv:"
            f"out_range=pc")


class Sources:
    def __init__(self, clip_dir, out_size):
        self.src = clip_dir / "source.mkv"
        self.info = media.probe(self.src)
        self.sq = media.square_size(self.info["w"], self.info["h"], self.info["sar"])
        self.out = out_size
        self.ref = clip_dir / "reference.mkv" if (clip_dir / "reference.mkv").exists() else None

    def lanczos(self):
        return media.to_rgb_filter(self.info, self.out, flags="lanczos")

    def square(self):
        return media.to_rgb_filter(self.info, self.sq, flags="lanczos")


def luma(x):
    return x @ LUMA


def small(y, k=8):
    h, w = (y.shape[0] // k) * k, (y.shape[1] // k) * k
    return y[:h, :w].reshape(h // k, k, w // k, k).mean(axis=(1, 3))


def psnr(a, b):
    mse = float(np.mean((a - b) ** 2))
    return 99.0 if mse <= 1e-12 else 10 * math.log10(1.0 / mse)


def ssim(a, b):
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    blur = lambda x: cv2.GaussianBlur(x, (11, 11), 1.5)  # noqa: E731
    ma, mb = blur(a), blur(b)
    va, vb, cov = blur(a * a) - ma * ma, blur(b * b) - mb * mb, blur(a * b) - ma * mb
    return float(np.mean(((2 * ma * mb + c1) * (2 * cov + c2)) / ((ma * ma + mb * mb + c1) * (va + vb + c2))))


def temporal_series(path, vf, size, flow_from=None):
    """Per-frame |Y_t - Y_t-1| (8x downscaled and full res); optionally warp error."""
    lp, hf, warp = [], [], []
    prev = None
    flows = iter(flow_from) if flow_from is not None else None
    for x in frames(path, vf, size):
        y = luma(x)
        if prev is not None:
            lp.append(float(np.mean(np.abs(small(y) - small(prev)))))
            hf.append(float(np.mean(np.abs(y - prev))))
            if flows is not None:
                fl = next(flows)
                warp.append(float(np.mean(np.abs(cv2.remap(prev, fl[0], fl[1], cv2.INTER_LINEAR,
                                                            borderMode=cv2.BORDER_REPLICATE) - y))))
        prev = y
    return np.array(lp), np.array(hf), np.array(warp) if flows is not None else None


def source_flows(srcs):
    """Backward flow (t -> t-1) of the Lanczos-upscaled source, as remap grids."""
    grids, prev = [], None
    w, h = srcs.out
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    for x in frames(srcs.src, srcs.lanczos(), srcs.out):
        g = (luma(x) * 255).astype(np.uint8)
        if prev is not None:
            half = cv2.calcOpticalFlowFarneback(cv2.resize(g, (w // 2, h // 2)), cv2.resize(prev, (w // 2, h // 2)),
                                                None, 0.5, 4, 21, 3, 5, 1.1, 0)
            f = cv2.resize(half, (w, h)) * 2
            grids.append((gx + f[..., 0], gy + f[..., 1]))
        prev = g
    return grids


def seams(run_dir, first_frame):
    """Frames (clip-relative) where a run switches batch, piece or job, from its plan."""
    p = run_dir / "plan.json"
    if not p.exists():
        return []
    pl = json.loads(p.read_text())
    timing = json.loads((run_dir / "timing.json").read_text()) if (run_dir / "timing.json").exists() else {}
    st = timing.get("settings", {})
    overlap, pre = st.get("overlap", 0), st.get("prepend_frames", 0)
    stats = [pc for j in timing.get("per_job", []) for pc in j.get("pieces", [])]
    out, k = [], 0
    for ji, job in enumerate(pl["jobs"]):
        for pi, pc in enumerate(job):
            if ji or pi:
                out.append(pc["keep_start"] + pc["blend_in"] // 2)
            if "batch_size" in st:
                b = stats[k].get("batch_size") if k < len(stats) else None
                b = b or plan.balanced_batch(pc["end"] - pc["start"] + pre, st["batch_size"], overlap)
                for s in plan.batch_starts(pc["end"] - pc["start"] + pre, b, overlap)[1:]:
                    out.append(pc["start"] + s - pre + overlap // 2)
            k += 1
    base = pl["first"]
    return sorted({f - base for f in out if pl["first"] < f < pl["last"]})


def svg_chart(series, marks, cuts, title, width=900, height=220):
    n = max((len(s) for _, s, _ in series), default=0)
    if not n:
        return ""
    vals = np.concatenate([s for _, s, _ in series])
    top = float(np.percentile(vals, 99.5)) if len(vals) else 1.0
    bot = min(0.0, float(np.percentile(vals, 0.5)))
    top = top if top > bot else bot + 1e-6
    X = lambda i: 40 + (width - 50) * i / max(1, n - 1)  # noqa: E731
    Y = lambda v: 10 + (height - 40) * (1 - (min(max(v, bot), top) - bot) / (top - bot))  # noqa: E731
    parts = [f'<svg viewBox="0 0 {width} {height}" width="100%" style="max-width:{width}px">',
             f'<text x="40" y="{height - 5}" font-size="11">{html.escape(title)} '
             f'(frame 1-{n}; ticks: seams per candidate, grey bars: scene cuts)</text>',
             f'<line x1="40" y1="{Y(0)}" x2="{width - 10}" y2="{Y(0)}" stroke="#bbb"/>']
    for c in cuts:
        if 0 < c <= n:
            parts.append(f'<rect x="{X(c - 1) - 1}" y="10" width="2" height="{height - 40}" fill="#ddd"/>')
    for k, (name, s, colour) in enumerate(series):
        pts = " ".join(f"{X(i):.1f},{Y(v):.1f}" for i, v in enumerate(s))
        parts.append(f'<polyline fill="none" stroke="{colour}" stroke-width="1.2" points="{pts}"><title>'
                     f'{html.escape(name)}</title></polyline>')
        for m in marks.get(name, []):
            if 0 < m <= n:
                y0 = height - 30 + 3 * (k % 4)
                parts.append(f'<line x1="{X(m - 1):.1f}" y1="{y0}" x2="{X(m - 1):.1f}" y2="{y0 + 3}" '
                             f'stroke="{colour}" stroke-width="2"/>')
    parts.append(f'<text x="2" y="{Y(top) + 4}" font-size="10">{top:.4f}</text>')
    parts.append("</svg>")
    return "\n".join(parts)


HEADER = 28


def labelled(img, text):
    """Tile with a black header strip carrying its name (never covering pixels)."""
    out = np.zeros((img.shape[0] + HEADER, img.shape[1], 3), np.uint8)
    out[HEADER:] = img
    cv2.putText(out, text, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def grab(path, vf, size, index):
    for i, x in enumerate(frames(path, vf + f",select=eq(n\\,{index})", size)):
        return x
    return None


def sheet(tiles, cols):
    h, w = tiles[0][1].shape[:2]
    h += HEADER
    rows = math.ceil(len(tiles) / cols)
    out = np.zeros((rows * h, cols * w, 3), np.uint8)
    for k, (name, t) in enumerate(tiles):
        r, c = divmod(k, cols)
        out[r * h:(r + 1) * h, c * w:(c + 1) * w] = labelled(
            (np.clip(t, 0, 1) * 255 + 0.5).astype(np.uint8)[..., ::-1], name)
    return out


def grid_video(inputs, size, dest, crop=None, max_width=3840):
    """inputs: [(name, path, vf)]; hstack/xstack them with labels."""
    n = len(inputs)
    cols = n if n <= 3 else math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    w, h = (crop[2] * 2, crop[3] * 2) if crop else size
    scale = min(1.0, max_width / (cols * w))
    tw, th = media.even(w * scale), media.even(h * scale)
    for labels in (True, False):
        args, fc = [], []
        for k, (name, path, vf) in enumerate(inputs):
            args += ["-i", path]
            chain = vf
            if crop:
                chain += f",crop={crop[2]}:{crop[3]}:{crop[0]}:{crop[1]},scale={crop[2] * 2}:{crop[3] * 2}:flags=neighbor"
            chain += f",scale={tw}:{th}:flags=lanczos,format=yuv420p,setsar=1"
            if labels:
                safe = name.replace("'", "").replace(":", " ")
                chain += (f",drawtext=text='{safe}':x=8:y=8:fontsize={max(14, th // 30)}:fontcolor=white:"
                          f"box=1:boxcolor=black@0.6")
            fc.append(f"[{k}:v]{chain}[v{k}]")
        for k in range(n, rows * cols):  # pad the grid with black tiles
            fc.append(f"color=black:s={tw}x{th}:d=1,format=yuv420p[v{k}]")
        layout = "|".join(f"{(k % cols) * tw}_{(k // cols) * th}" for k in range(rows * cols))
        fc.append("".join(f"[v{k}]" for k in range(rows * cols)) +
                  (f"xstack=inputs={rows * cols}:layout={layout}:shortest=1[out]" if rows * cols > 1 else "null[out]"))
        r = ffmpeg(*args, "-filter_complex", ";".join(fc), "-map", "[out]", "-c:v", "libx264", "-crf", "16",
                   "-preset", "medium", "-pix_fmt", "yuv420p", dest)
        if r.returncode == 0:
            return True
    print(f"warning: could not make {dest}: {r.stderr.decode()[-500:]}", flush=True)
    return False


def fmt(v, spec=".3f"):
    return "-" if v is None or (isinstance(v, float) and math.isnan(v)) else format(v, spec)


def report_clip(clip_dir, ev, a):
    meta = json.loads((clip_dir / "clip.json").read_text())
    runs = []
    for rj in sorted((ev / "runs" / meta["name"]).glob("*/run.json")):
        r = json.loads(rj.read_text())
        r["dir"] = rj.parent
        runs.append(r)
    if not runs:
        return f"<h2>{html.escape(meta['name'])}</h2><p>no runs yet</p>"
    order = {c: i for i, c in enumerate(a.order)}
    runs.sort(key=lambda r: (order.get(r["candidate"]["name"], 99), r["candidate"]["name"]))
    size = tuple(runs[0]["size"])
    srcs = Sources(clip_dir, size)
    rep = ev / "report" / meta["name"]
    rep.mkdir(parents=True, exist_ok=True)
    print(f"clip {meta['name']}: {len(runs)} runs at {size[0]}x{size[1]}", flush=True)
    crop = [int(v) for v in meta["crop"].split(",")] if meta.get("crop") else None

    names = [r["candidate"]["name"] for r in runs]
    inputs = [("source (lanczos)", srcs.src, srcs.lanczos())]
    if srcs.ref:
        inputs.append(("original (reference)", srcs.ref, f"scale={size[0]}:{size[1]}:flags=lanczos"))
    inputs += [(n, r["dir"] / "output.mkv", out_filter(size)) for n, r in zip(names, runs)]
    grid_video(inputs, size, rep / "grid.mp4")
    if crop:
        grid_video(inputs, size, rep / "crops.mp4", crop=crop)

    # Stills: region of interest (or the centre) at 2x pixel zoom, every input.
    n_frames = meta["frames"]
    cx, cy, cw, ch = crop or (size[0] // 2 - 240, size[1] // 2 - 135, 480, 270)
    stills = []
    for k, pos in enumerate(a.stills):
        idx = min(n_frames - 1, int(pos * n_frames))
        tiles = []
        for name, path, vf in inputs:
            x = grab(path, vf, size, idx)
            if x is not None:
                z = cv2.resize(x[cy:cy + ch, cx:cx + cw], (cw * 2, ch * 2), interpolation=cv2.INTER_NEAREST)
                tiles.append((name, z))
        if tiles:
            cv2.imwrite(str(rep / f"still{k}.png"), sheet(tiles, min(3, len(tiles))))
            stills.append((f"still{k}.png", idx))

    # x-t slice of one row (within the region of interest's columns)
    row = cy + ch // 2 if crop else int(meta.get("row", 0.5) * size[1])
    slices = []
    for name, path, vf in inputs:
        rows = [x[row, cx:cx + cw] for x in frames(path, vf, size)]
        if rows:
            slices.append((name, np.stack(rows)))
    if slices:
        t = max(s.shape[0] for _, s in slices)
        slices = [(nm, np.pad(s, ((0, t - s.shape[0]), (0, 0), (0, 0)))) for nm, s in slices]
        cv2.imwrite(str(rep / "xt.png"), sheet(slices, len(slices)))

    # Metrics
    flows = source_flows(srcs) if a.flow else None
    lz_lp, lz_hf, lz_warp = temporal_series(srcs.src, srcs.lanczos(), size, flows)
    src_frames = list(frames(srcs.src, srcs.square(), srcs.sq))
    # Degraded clips: compare with the original at the original's size (never upscale it).
    ref_size = None
    if srcs.ref:
        ri = media.probe(srcs.ref)
        ref_size = (ri["w"], ri["h"]) if ri["h"] <= size[1] else size
        ref_frames = list(frames(srcs.ref, f"scale={ref_size[0]}:{ref_size[1]}:flags=area", ref_size))
    else:
        ref_frames = None
    cut_list = []
    first_plan = next((r["dir"] / "plan.json" for r in runs if (r["dir"] / "plan.json").exists()), None)
    if first_plan:
        pl = json.loads(first_plan.read_text())
        cut_list = [c - pl["first"] for c in pl["scene_cuts"]]
    excl = set()
    for c in cut_list:
        excl.update(range(c - 1, c + 2))
    rows_html, series_lp, marks = [], [], {}
    for k, (name, r) in enumerate(zip(names, runs)):
        out = r["dir"] / "output.mkv"
        lp, hf, warp = temporal_series(out, out_filter(size), size, flows)
        m = min(len(lp), len(lz_lp))
        ex_lp, ex_hf = lp[:m] - lz_lp[:m], hf[:m] - lz_hf[:m]
        keep = np.array([i + 1 not in excl for i in range(m)], bool)
        sm = seams(r["dir"], meta["first_frame"])
        marks[name] = sm
        # Seam jump: the peak flicker+ around each of the run's seams, minus the typical
        # (median) flicker+ elsewhere, averaged over seams. In 8-bit steps.
        near = np.zeros(m, bool)
        peaks = []
        for s in sm:
            lo, hi = max(0, s - 3), min(m, s + 3)
            w = [i for i in range(lo, hi) if keep[i]]
            near[lo:hi] = True
            if w:
                peaks.append(max(ex_lp[i] for i in w))
        rest = ex_lp[keep & ~near]
        seam_jump = (float(np.mean(peaks)) - float(np.median(rest))) * 255 if peaks and len(rest) else None
        fid, shift, fr_psnr, fr_ssim = [], [], [], []
        for i, x in enumerate(frames(out, out_filter(srcs.sq, "area"), srcs.sq)):
            if i < len(src_frames):
                fid.append(psnr(luma(x), luma(src_frames[i])))
                shift.append((x.mean(axis=(0, 1)) - src_frames[i].mean(axis=(0, 1))) * 255)
        if ref_frames:
            for i, x in enumerate(frames(out, out_filter(ref_size, "area"), ref_size)):
                if i < len(ref_frames):
                    fr_psnr.append(psnr(luma(x), luma(ref_frames[i])))
                    fr_ssim.append(ssim(luma(x), luma(ref_frames[i])))
        warp_ex = float(np.mean(warp[:m] - lz_warp[:m])) if warp is not None and len(warp) else None
        sh = np.mean(shift, axis=0) if shift else [float("nan")] * 3
        series_lp.append((name, ex_lp, COLOURS[(k + 1) % len(COLOURS)]))
        cand = r["candidate"]
        what = cand.get("args") or cand.get("command") or cand.get("file")
        rows_html.append(
            "<tr>" + "".join(f"<td>{v}</td>" for v in [
                f'<b style="color:{COLOURS[(k + 1) % len(COLOURS)]}">{html.escape(name)}</b>',
                html.escape(cand.get("why", "")) + f"<br><code>{html.escape(json.dumps(what))}</code>",
                fmt(r.get("wall_seconds"), ".0f"), fmt(r.get("fps"), ".3f"),
                fmt(r.get("gpu_peak_mib") and r["gpu_peak_mib"] / 1024, ".1f"),
                "ok" if r["frames"] == r["expected_frames"] else f"<b>{r['frames']}/{r['expected_frames']}</b>",
                fmt(float(np.mean(ex_lp[keep])) * 255 if keep.any() else None, ".3f"),
                fmt(float(np.percentile(ex_lp[keep], 99)) * 255 if keep.any() else None, ".3f"),
                fmt(float(np.mean(ex_hf[keep])) * 255 if keep.any() else None, ".3f"),
                fmt(seam_jump, ".2f"), fmt(float(np.mean(fid)) if fid else None, ".2f"),
                " / ".join(fmt(float(v), "+.2f") for v in sh),
                fmt(float(np.mean(fr_psnr)) if fr_psnr else None, ".2f"),
                fmt(float(np.mean(fr_ssim)) if fr_ssim else None, ".4f"),
                fmt(warp_ex * 255 if warp_ex is not None else None, ".3f")]) + "</tr>")
        (rep / f"metrics_{name}.json").write_text(json.dumps(dict(
            excess_flicker_lp=ex_lp.tolist(), excess_flicker_hf=ex_hf.tolist(), seams=sm,
            fidelity_psnr=fid, ref_psnr=fr_psnr, ref_ssim=fr_ssim), indent=0))

    head = ["candidate", "what", "wall s", "fps", "GPU peak GiB", "frames", "flicker+ lp (mean, 8-bit)",
            "flicker+ lp p99", "flicker+ full res", "seam jump", "fidelity PSNR dB", "colour shift R/G/B",
            "ref PSNR dB", "ref SSIM", "warp err+"]
    info = srcs.info
    parts = [f"<h2 id=\"{html.escape(meta['name'])}\">{html.escape(meta['name'])}</h2>",
             f"<p>{meta['frames']} frames from <code>{html.escape(Path(meta['origin']).name)}</code> at frame "
             f"{meta['first_frame']}; source {info['w']}x{info['h']} SAR {info['sar']} ({info['matrix']}); "
             f"output {size[0]}x{size[1]}; tags: {html.escape(', '.join(meta['tags']) or '-')}; "
             f"{'degraded from the reference with ' + meta['degrade'] if meta.get('degrade') else 'real footage'}. "
             f"{html.escape(meta.get('notes') or '')}</p>",
             f'<p><a href="{meta["name"]}/grid.mp4">grid.mp4</a>'
             + (f' · <a href="{meta["name"]}/crops.mp4">crops.mp4</a>' if crop else "") + "</p>",
             "<table><tr>" + "".join(f"<th>{h}</th>" for h in head) + "</tr>" + "".join(rows_html) + "</table>",
             svg_chart(series_lp, marks, cut_list, "flicker+ (8x downscaled luma) per frame")]
    for f, idx in stills:
        parts.append(f'<figure><img src="{meta["name"]}/{f}" loading="lazy"><figcaption>frame {idx + 1}, '
                     f'region {cw}x{ch} at {cx},{cy}, 2x pixel zoom</figcaption></figure>')
    if slices:
        parts.append(f'<figure><img src="{meta["name"]}/xt.png" loading="lazy"><figcaption>x-t slice of row '
                     f'{row} (time runs down)</figcaption></figure>')
    return "\n".join(parts)


PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>Restoration comparison</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>body{{font:14px system-ui,sans-serif;margin:16px;max-width:1800px}}
table{{border-collapse:collapse;margin:8px 0}}td,th{{border:1px solid #ccc;padding:3px 6px;vertical-align:top}}
th{{background:#f2f2f2;font-weight:600}}code{{font-size:11px}}img{{max-width:100%}}
figure{{margin:12px 0}}figcaption{{color:#555}}</style></head><body>
<h1>Restoration comparison</h1>
<p>Generated {when}. Watch the grid videos first; the numbers only say where to look.
Column meanings are in <code>eval/report.py</code>.</p>
{body}
</body></html>"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval", required=True)
    ap.add_argument("--clips")
    ap.add_argument("--flow", action="store_true", help="optical-flow warp error (slow)")
    ap.add_argument("--stills", default="0.2,0.5,0.8", help="positions (0-1) of the still frames")
    ap.add_argument("--order", default="lanczos", help="candidates listed first, comma-separated")
    a = ap.parse_args()
    a.stills = [float(s) for s in a.stills.split(",")]
    a.order = a.order.split(",")
    ev = Path(a.eval)
    clips = sorted(p.parent for p in (ev / "clips").glob("*/clip.json"))
    if a.clips:
        clips = [c for c in clips if c.name in a.clips.split(",")]
    body = [report_clip(c, ev, a) for c in clips]
    import time
    (ev / "report").mkdir(exist_ok=True)
    (ev / "report" / "index.html").write_text(PAGE.format(when=time.strftime("%Y-%m-%d %H:%M"),
                                                          body="\n".join(body)))
    print(f"report: {ev / 'report' / 'index.html'}", flush=True)


if __name__ == "__main__":
    main()
