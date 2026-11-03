"""Split a frame range into shots, pieces and jobs. Pure functions, no I/O.

Vocabulary
  shot   frames between two scene cuts. Nothing is ever restored across a cut:
         temporal attention and per-batch colour statistics would mix two
         unrelated shots.
  piece  one call of the restoration model: a shot, or part of a long shot.
         Pieces of the same long shot overlap by `overlap` frames.
  job    consecutive pieces handled by one worker (one GPU) in one go, then
         encoded to one video file. Jobs end at shot cuts unless a single shot
         is longer than a job.

Piece fields (all absolute source frame numbers, end-exclusive):
  start, end            frames the model reads
  keep_start, keep_end  frames this piece contributes to the output
  blend_in              its first `blend_in` kept frames are crossfaded with
                        the previous piece's version of the same frames
  blend_out             its last `blend_out` kept frames are not written; they
                        are handed to the next piece (same job) for that crossfade
Overlapping pieces in the same job are crossfaded; overlapping pieces in
different jobs are cut in the middle of the overlap instead (each still has
context beyond the cut).
"""
from math import ceil


def shots(first, last, cuts):
    """[first, last) split at every cut strictly inside it."""
    bounds = [first] + sorted(c for c in set(cuts) if first < c < last) + [last]
    return list(zip(bounds[:-1], bounds[1:]))


def split_shot(a, b, max_len, overlap):
    """Read ranges covering [a, b), at most max_len long, consecutive ones
    overlapping by exactly `overlap` frames, sizes as even as possible."""
    if not 0 <= overlap < max_len:
        raise ValueError("need 0 <= overlap < max_len")
    n_frames = b - a
    if n_frames <= max_len:
        return [(a, b)]
    n = ceil((n_frames - overlap) / (max_len - overlap))
    step = (n_frames - overlap) / n
    starts = [a + round(j * step) for j in range(n)]
    ends = [starts[j + 1] + overlap for j in range(n - 1)] + [b]
    return list(zip(starts, ends))


def balanced_batch(n_frames, max_batch, overlap):
    """Batch size (<= max_batch) that splits n_frames into the same number of
    batches as max_batch would, but of near-equal length, instead of leaving a
    short last batch with little temporal context. Mirrors how the SeedVR2 batch
    loop steps (batch - overlap frames per batch).

    A 4n+1 size is preferred (no padding). When rounding to 4n+1 would bring the
    short last batch back, the exact size is used: SeedVR2 pads every batch to
    4n+1 with mirrored frames, which costs at most 3 frames of compute per batch."""
    if (max_batch - 1) % 4:
        raise ValueError("max_batch must be 4n+1")
    if n_frames <= max_batch or overlap >= max_batch:
        return max_batch
    k = ceil((n_frames - overlap) / (max_batch - overlap))
    exact = ceil((n_frames - overlap) / k) + overlap
    b4 = min(max_batch, exact + (-(exact - 1)) % 4)
    last = n_frames - batch_starts(n_frames, b4, overlap)[-1]
    return b4 if last - overlap >= (b4 - overlap) // 2 else exact


def batch_starts(n_frames, batch, overlap):
    """Start offsets of the SeedVR2 batches for a piece (same loop as upstream)."""
    step = batch - overlap if 0 < overlap < batch else batch
    starts = []
    for s in range(0, n_frames, step):
        if s > 0 and min(s + batch, n_frames) - s <= (overlap if overlap < batch else 0):
            break
        starts.append(s)
    return starts


def _piece(start, end):
    return dict(start=start, end=end, keep_start=start, keep_end=end, blend_in=0, blend_out=0)


def plan_jobs(first, last, cuts, job_frames, max_piece, overlap):
    """List of jobs, each a list of piece dicts (see module docstring)."""
    if job_frames < max_piece:
        raise ValueError("job_frames must be >= max_piece")
    if not 0 <= 4 * overlap <= max_piece:
        raise ValueError("overlap must be at most a quarter of max_piece")  # room for both crossfades
    jobs, cur, cur_len = [], [], 0
    for shot_no, (a, b) in enumerate(shots(first, last, cuts)):
        pieces = [_piece(s, e) for s, e in split_shot(a, b, max_piece, overlap)]
        for p in pieces:
            p["shot"] = shot_no
        for prev, nxt in zip(pieces, pieces[1:]):
            prev["blend_out"] = nxt["blend_in"] = overlap
        if cur and cur_len + (b - a) > job_frames:
            jobs.append(cur)
            cur, cur_len = [], 0
        for p in pieces:
            own = p["end"] - p["start"] - p["blend_in"]  # frames not shared with the previous piece
            if cur and cur_len + own > job_frames:
                if cur[-1]["shot"] == shot_no:
                    _cut_between(cur[-1], p)
                jobs.append(cur)
                cur, cur_len = [], 0
            cur.append(p)
            cur_len += p["keep_end"] - p["keep_start"] - p["blend_in"]
    if cur:
        jobs.append(cur)
    return jobs


def _cut_between(prev, nxt):
    """Overlapping pieces that land in different jobs: hard cut mid-overlap."""
    ov = prev["end"] - nxt["start"]
    mid = nxt["start"] + ov // 2
    prev["keep_end"], prev["blend_out"] = mid, 0
    nxt["keep_start"], nxt["blend_in"] = mid, 0


def job_range(job):
    return job[0]["keep_start"], job[-1]["keep_end"]


def written_frames(piece):
    """Frames a piece writes into the job's output (its blend_out tail is
    written later by the next piece)."""
    return range(piece["keep_start"], piece["keep_end"] - piece["blend_out"])


def check(jobs, first, last, cuts):
    """Raise ValueError unless the plan writes every frame of [first, last)
    exactly once and never reads across a cut. Used by tests and at runtime."""
    seen = []
    cutset = set(cuts)
    for job in jobs:
        for prev, nxt in zip(job, job[1:]):
            if prev["blend_out"] != nxt["blend_in"]:
                raise ValueError(f"blend mismatch {prev} {nxt}")
            if prev["blend_out"] and prev["keep_end"] - prev["blend_out"] != nxt["keep_start"]:
                raise ValueError(f"blend frames misaligned {prev} {nxt}")
        if job[0]["blend_in"] or job[-1]["blend_out"]:
            raise ValueError("a crossfade crosses a job boundary")
        for p in job:
            if not (first <= p["start"] <= p["keep_start"] < p["keep_end"] <= p["end"] <= last):
                raise ValueError(f"bad piece {p}")
            if p["keep_end"] - p["keep_start"] < p["blend_in"] + p["blend_out"]:
                raise ValueError(f"piece too short for its crossfades {p}")
            if any(p["start"] < c < p["end"] for c in cutset):
                raise ValueError(f"piece crosses a scene cut {p}")
            seen.extend(written_frames(p))
    if seen != list(range(first, last)):
        raise ValueError("plan does not cover the range exactly once, in order")
