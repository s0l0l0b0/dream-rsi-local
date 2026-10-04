"""Small shared helpers used across the Dream-RSI codebase.

Kept dependency-free and side-effect-light so that every other module (including
the sandboxed evaluator) can import it safely.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping


# --------------------------------------------------------------------------------------
# Time / ids
# --------------------------------------------------------------------------------------
def utc_now() -> str:
    """Return an ISO-8601 UTC timestamp with second precision."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def new_id(prefix: str, length: int = 8) -> str:
    """Return a short, collision-resistant identifier such as ``node_3f9a2c10``."""
    return f"{prefix}_{uuid.uuid4().hex[:length]}"


def monotonic_ms() -> float:
    """Monotonic clock in milliseconds (for latency measurement)."""
    return time.perf_counter() * 1000.0


# --------------------------------------------------------------------------------------
# Hashing / JSON
# --------------------------------------------------------------------------------------
def stable_hash(obj: Any, length: int = 16) -> str:
    """Deterministic short hash of an arbitrary JSON-serializable object.

    Used for action signatures, evaluator integrity checks and cache keys, so it
    must never depend on dict ordering or platform-specific repr.
    """
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:length]


def file_sha256(path: str | os.PathLike[str]) -> str:
    """SHA-256 of a file's bytes, used to detect tampering with frozen artifacts."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def seeded_int(*parts: Any) -> int:
    """A 32-bit seed derived from ``parts`` that is stable **across processes**.

    ``hash()`` must never be used for policy decisions in this project: CPython
    randomises string hashing per process, and the Dream Engine replays policies
    in a *different* process than the live explorer, so a ``hash()``-derived
    choice would silently differ between exploration and replay.
    """
    return int(stable_hash(list(parts), 8), 16)


def to_jsonable(obj: Any) -> Any:
    """Recursively convert ``obj`` into plain JSON-serializable Python objects."""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, Mapping):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_jsonable(v) for v in obj]
    if hasattr(obj, "to_dict"):
        return to_jsonable(obj.to_dict())
    return str(obj)


def dumps(obj: Any, *, indent: int | None = 2) -> str:
    """JSON dump with deterministic key ordering and numpy/Path-safe defaults."""
    return json.dumps(to_jsonable(obj), indent=indent, sort_keys=False, ensure_ascii=False)


def atomic_write_text(path: str | os.PathLike[str], text: str) -> None:
    """Write ``text`` to ``path`` atomically (temp file + ``os.replace``)."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=str(target.parent), delete=False, suffix=".tmp"
    )
    try:
        with handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(handle.name, target)
    except BaseException:
        Path(handle.name).unlink(missing_ok=True)
        raise


def read_json(path: str | os.PathLike[str], default: Any = None) -> Any:
    """Read JSON, returning ``default`` when the file is absent."""
    p = Path(path)
    if not p.exists():
        return default
    with p.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: str | os.PathLike[str], obj: Any, *, indent: int | None = 2) -> None:
    """Atomically write ``obj`` as pretty JSON."""
    atomic_write_text(path, dumps(obj, indent=indent) + "\n")


# --------------------------------------------------------------------------------------
# Statistics (stdlib only; no numpy dependency at runtime)
# --------------------------------------------------------------------------------------
def mean(values: Iterable[float]) -> float:
    """Arithmetic mean, ``0.0`` for an empty sequence."""
    items = list(values)
    return float(sum(items) / len(items)) if items else 0.0


def stdev(values: Iterable[float]) -> float:
    """Population standard deviation, ``0.0`` for fewer than two samples."""
    items = list(values)
    if len(items) < 2:
        return 0.0
    mu = mean(items)
    return float((sum((v - mu) ** 2 for v in items) / len(items)) ** 0.5)


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    """Clamp ``value`` into ``[low, high]``."""
    return max(low, min(high, value))


def percentile(values: Iterable[float], q: float) -> float:
    """Linear-interpolated percentile (``q`` in ``[0, 1]``)."""
    items = sorted(values)
    if not items:
        return 0.0
    if len(items) == 1:
        return float(items[0])
    pos = clamp(q) * (len(items) - 1)
    low = int(pos)
    high = min(low + 1, len(items) - 1)
    frac = pos - low
    return float(items[low] * (1 - frac) + items[high] * frac)


def safe_div(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Division that returns ``default`` instead of raising on a zero denominator."""
    return float(numerator / denominator) if denominator else default
