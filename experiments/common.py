"""Small helpers shared by the experiment scripts: package paths, unit parsing, JSON/CSV writers,
and the environment record stored in every ``manifest.json``.

Nothing in this module touches a random number generator or the numerics of an experiment
(``restart_mean`` only re-derives, for the rescoring checks, the restart aggregation of
``ssrk.selfrecon.ssrk_scores``), and it imports neither torch nor scikit-learn (the CPU-only Level A
scripts depend on it).
"""
from __future__ import annotations

import csv
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

#: Package root (the directory that contains ``ssrk/`` and ``experiments/``).
PKG_ROOT = Path(__file__).resolve().parents[1]
#: Default data directory (``data/pbmc/...``, ``data/uci_har/UCI HAR Dataset/...``).
DEFAULT_DATA_DIR = PKG_ROOT / "data"
#: Default results directory (``results/<area>/...``).
DEFAULT_RESULTS_DIR = PKG_ROOT / "results"
#: Expected outputs copied from the original research records.
REFERENCE_DIR = PKG_ROOT / "reference"


def resolve_path(path: str | os.PathLike | None, default: Path) -> Path:
    """Resolve a CLI path; relative paths are taken relative to the package root."""
    if path is None:
        return default
    p = Path(path)
    return p if p.is_absolute() else (PKG_ROOT / p)


def display_path(path: str | os.PathLike) -> str:
    """``path`` relative to the package root if it lies inside it (for reports), else as given."""
    p = Path(path)
    try:
        return p.resolve().relative_to(PKG_ROOT.resolve()).as_posix()
    except ValueError:
        return str(p)


def parse_units(text: str, n_units: int) -> list[int]:
    """Parse ``"all"``, ``"0,5"`` or ``"0-4,7"`` into a sorted list of unit indices."""
    text = str(text).strip().lower()
    if text in {"all", ""}:
        return list(range(n_units))
    units: set[int] = set()
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = (int(v) for v in part.split("-", 1))
            units.update(range(lo, hi + 1))
        else:
            units.add(int(part))
    bad = [u for u in units if not 0 <= u < n_units]
    if bad:
        raise ValueError(f"unit indices {bad} outside 0..{n_units - 1}")
    return sorted(units)


def json_ready(value: Any) -> Any:
    """Convert numpy scalars/arrays, sets and paths to JSON-serializable Python objects."""
    try:
        import numpy as np
    except Exception:  # pragma: no cover
        np = None
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, set):
        return sorted(json_ready(v) for v in value)
    if isinstance(value, Path):
        return str(value)
    if np is not None:
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            return float(value)
        if isinstance(value, np.bool_):
            return bool(value)
    return value


def save_json(path: Path, payload: Any, indent: int | None = 1) -> None:
    """Write ``payload`` as JSON (UTF-8), creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(payload), indent=indent), encoding="utf-8")


def load_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_csv(path: Path, rows: Sequence[dict], fields: Sequence[str]) -> None:
    """Write ``rows`` (dicts) to ``path`` with the given column order."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(json_ready(row))


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def environment_record() -> dict[str, Any]:
    """Versions of the numerical stack and the device; stored in every manifest.

    torch is inspected only if the calling script has already imported it (the training scripts).
    The CPU-only scripts (rescoring, stability, evaluator) never import torch; for them the record
    holds the installed torch version read from the package metadata, or ``"not installed"``.
    """
    env: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor(),
    }
    from importlib import metadata

    for name, dist in (("numpy", "numpy"), ("scipy", "scipy"), ("sklearn", "scikit-learn"), ("pandas", "pandas"),
                       ("anndata", "anndata"), ("h5py", "h5py"), ("threadpoolctl", "threadpoolctl")):
        try:
            env[name] = metadata.version(dist)
        except Exception:
            env[name] = "not installed"
    torch = sys.modules.get("torch")
    if torch is not None and hasattr(torch, "cuda"):
        env["torch"] = torch.__version__
        env["cuda_available"] = bool(torch.cuda.is_available())
        env["cuda"] = torch.version.cuda
        env["cudnn"] = torch.backends.cudnn.version() if torch.cuda.is_available() else None
        env["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    else:
        try:
            env["torch"] = metadata.version("torch") + " (installed, not imported)"
        except Exception:
            env["torch"] = "not installed"
    env["CUBLAS_WORKSPACE_CONFIG"] = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    return env


def command_line() -> str:
    """The command used to launch the current script (for manifests)."""
    return " ".join([Path(sys.executable).name] + [str(a) for a in sys.argv])


def write_manifest(out_dir: Path, payload: dict[str, Any], name: str = "manifest.json") -> Path:
    """Write ``out_dir/<name>`` with the command, environment and the given payload."""
    record = {"command": command_line(), "created_utc": utc_now(), "environment": environment_record()}
    record.update(payload)
    path = Path(out_dir) / name
    if path.exists() and "unit_seconds" in record:
        # keep the per-unit timings of earlier invocations that wrote into the same directory
        try:
            previous = load_json(path).get("unit_seconds", {})
        except Exception:
            previous = {}
        record["unit_seconds"] = {**previous, **record["unit_seconds"]}
    save_json(path, record)
    return path


def blas_threads(n: int):
    """Context limiting BLAS/OpenMP threads to ``n`` (``threadpoolctl``); ``n <= 0`` keeps library defaults.

    The scikit-learn evaluators (randomized PCA, k-means, MLP) can depend on the thread count in rare
    cases; single-thread evaluation makes them independent of the core count.
    """
    import contextlib

    if n is None or int(n) <= 0:
        return contextlib.nullcontext()
    from threadpoolctl import threadpool_limits

    return threadpool_limits(limits=int(n))


def compare_unit_with_reference(units_dir: Path, ref_dir: Path, unit: int, keys: Sequence[str]) -> dict[str, bool]:
    """Bitwise comparison of one unit's outputs with a reference record set.

    Compares ``NNN_W.npy`` and ``NNN_Wall.npy`` (dtype, shape and bytes) and the given JSON fields
    (exact equality of the parsed values). Returns ``{name: equal}``.
    """
    import numpy as np

    out: dict[str, bool] = {}
    for name in ("W", "Wall"):
        out[name] = same_array(np.load(Path(units_dir) / f"{unit:03d}_{name}.npy"),
                               np.load(Path(ref_dir) / f"{unit:03d}_{name}.npy"))
    a = load_json(Path(units_dir) / f"{unit:03d}.json")
    b = load_json(Path(ref_dir) / f"{unit:03d}.json")
    for k in keys:
        out[k] = a.get(k) == b.get(k)
    return out


def sha256_file(path: Path) -> str:
    """SHA-256 hex digest of a file."""
    import hashlib

    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def same_array(x, y) -> bool:
    """``True`` if two numpy arrays have identical dtype, shape and bytes."""
    return bool(x.dtype == y.dtype and x.shape == y.shape and x.tobytes() == y.tobytes())


def restart_mean(Wall):
    """Restart aggregation of ``ssrk.selfrecon.ssrk_scores`` re-derived for the rescoring checks:
    coordinatewise float64 mean over the restarts (rows of ``Wall``), stored as float32."""
    import numpy as np

    return Wall.astype(np.float64).mean(axis=0).astype(np.float32)


def iter_chunks(items: Sequence[Any], n: int) -> Iterable[Sequence[Any]]:
    """Round-robin split ``items`` into ``n`` interleaved chunks (``items[w::n]``)."""
    for w in range(n):
        yield items[w::n]
