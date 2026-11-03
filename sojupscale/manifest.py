"""Resume safety: a work directory records everything that determines its pixels.

`identity` is a JSON-able dict (source content, range, sizes, colour handling,
plan and engine settings, engine/model versions, encode settings). It is stored
in WORK/manifest.json on the first run; a rerun with a different identity is
refused and the differing keys are listed, so finished pieces are never reused
for a different input, model or setting."""
import hashlib
import json
import os
from pathlib import Path

SAMPLE = 4 << 20  # bytes hashed at the start, middle and end of the source


def file_id(path):
    """Cheap content identity: size plus SHA-256 of three 4 MiB samples.
    Catches a replaced or re-encoded source without reading a 20 GB file."""
    path = Path(path)
    size = path.stat().st_size
    h = hashlib.sha256(str(size).encode())
    with open(path, "rb") as f:
        for off in sorted({0, max(0, size // 2 - SAMPLE // 2), max(0, size - SAMPLE)}):
            f.seek(off)
            h.update(f.read(SAMPLE))
    return f"{size}:{h.hexdigest()[:32]}"


def tree_hash(paths):
    """SHA-256 over the contents of the given files (sorted by name)."""
    h = hashlib.sha256()
    for p in sorted(Path(p) for p in paths):
        h.update(p.name.encode())
        h.update(p.read_bytes())
    return h.hexdigest()[:16]


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()[:16]


def _flatten(d, prefix=""):
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, key + "."))
        else:
            out[key] = v
    return out


def diff(old, new):
    a, b = _flatten(old), _flatten(new)
    return sorted((k, a.get(k), b.get(k)) for k in set(a) | set(b) if a.get(k) != b.get(k))


class Mismatch(Exception):
    pass


def ensure(work, identity):
    """Create WORK/manifest.json, or check it matches. Raises Mismatch."""
    work = Path(work)
    path = work / "manifest.json"
    if path.exists():
        old = json.loads(path.read_text())["identity"]
        changes = diff(old, identity)
        if changes:
            lines = "\n".join(f"  {k}: {a!r} -> {b!r}" for k, a, b in changes)
            raise Mismatch(f"{work} was made with different input/settings:\n{lines}\n"
                           "Use a new --work directory, or --restart to discard it.")
        return False
    if any(work.glob("seg[0-9][0-9][0-9]")) or (work / "segments.json").exists():
        raise Mismatch(f"{work} was made by an older version of upscale.py (no manifest); "
                       "use a new --work directory, or --restart to discard it.")
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(dict(identity=identity, digest=digest(identity)), indent=1))
    os.replace(tmp, path)
    return True
