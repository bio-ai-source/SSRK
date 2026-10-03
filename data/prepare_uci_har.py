#!/usr/bin/env python
"""Processing step for UCI HAR: extract a manually downloaded archive, verify it, optionally cache.

The UCI HAR dataset (UCI Machine Learning Repository id 240, doi:10.24432/C54S4K) is too large to
ship (282.6 MB extracted).  ``data/download.py --uci-har`` fetches and extracts it
automatically; this script covers the remaining cases:

1. **Manual download** (e.g. behind a firewall): download
   https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip
   in a browser, then run::

       python data/prepare_uci_har.py --zip path/to/human+activity+recognition+using+smartphones.zip

   Either the outer archive (which wraps ``UCI HAR Dataset.zip``) or the inner
   ``UCI HAR Dataset.zip`` is accepted.  The files read by the experiments are extracted to
   ``data/uci_har/UCI HAR Dataset/`` and verified against ``data/checksums.json``.

2. **Verification only** (no ``--zip``): verifies the extracted files.

3. **Optional array cache** (``--cache``): parses the text files once and writes
   ``data/uci_har/uci_har_arrays.npz``; every array is re-read and checked to be *bitwise equal*
   to the arrays produced by the loader the experiments use.

Processing performed by the experiments (and reproduced here for the cache)
----------------------------------------------------------------------------
No preprocessing beyond parsing: the official 7,352 / 2,947 train/test split and the 561 supplied
features (already scaled to [-1, 1] by the dataset creators) are used as is, without further
standardisation::

    X_train = np.loadtxt("train/X_train.txt", dtype=np.float32)      # (7352, 561)
    y_train = np.loadtxt("train/y_train.txt", dtype=int) - 1          # labels 0..5
    X_test  = np.loadtxt("test/X_test.txt",  dtype=np.float32)        # (2947, 561)
    y_test  = np.loadtxt("test/y_test.txt",  dtype=int) - 1

Feature names are the second whitespace-separated field of each line of ``features.txt``
(names starting with ``f`` are frequency-domain features; they define the domain-imbalance
metric).  **The experiments read the text files**, not the cache; the cache is a convenience for
interactive analysis and is never required.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import download as dl  # noqa: E402  (sibling module data/download.py)

DATASET_SUBDIR = Path("uci_har") / "UCI HAR Dataset"
CACHE_NAME = "uci_har_arrays.npz"


def load_uci_har_text(dataset_dir: Path) -> dict[str, np.ndarray]:
    """Parse the UCI HAR text files exactly as the experiments' loader does.

    Returns a dict with ``X_train``/``X_test`` (float32), ``y_train``/``y_test`` (0-based int),
    ``subject_train``/``subject_test`` (int, 1..30), ``feature_names`` (561 str) and
    ``activity_labels`` (6 str, index = 0-based label).
    """
    d = Path(dataset_dir)
    arrays = {
        "X_train": np.loadtxt(d / "train" / "X_train.txt", dtype=np.float32),
        "y_train": np.loadtxt(d / "train" / "y_train.txt", dtype=int) - 1,
        "X_test": np.loadtxt(d / "test" / "X_test.txt", dtype=np.float32),
        "y_test": np.loadtxt(d / "test" / "y_test.txt", dtype=int) - 1,
        "subject_train": np.loadtxt(d / "train" / "subject_train.txt", dtype=int),
        "subject_test": np.loadtxt(d / "test" / "subject_test.txt", dtype=int),
    }
    names = []
    for line in (d / "features.txt").read_text(encoding="utf-8").splitlines():
        parts = line.strip().split(maxsplit=1)
        if len(parts) == 2:
            names.append(parts[1])
    arrays["feature_names"] = np.array(names)
    labels = {}
    for line in (d / "activity_labels.txt").read_text(encoding="utf-8").splitlines():
        parts = line.strip().split(maxsplit=1)
        if len(parts) == 2:
            labels[int(parts[0]) - 1] = parts[1]
    arrays["activity_labels"] = np.array([labels[i] for i in sorted(labels)])
    return arrays


def check_shapes(a: dict[str, np.ndarray]) -> None:
    """Assert the documented shapes of the official split."""
    expected = {"X_train": (7352, 561), "X_test": (2947, 561), "y_train": (7352,),
                "y_test": (2947,), "subject_train": (7352,), "subject_test": (2947,),
                "feature_names": (561,), "activity_labels": (6,)}
    for key, shape in expected.items():
        if a[key].shape != shape:
            raise dl.DataError(f"{key} has shape {a[key].shape}, expected {shape}")
    for key in ("y_train", "y_test"):
        if set(np.unique(a[key]).tolist()) != set(range(6)):
            raise dl.DataError(f"{key} does not contain exactly the labels 0..5")


def bitwise_equal(a: np.ndarray, b: np.ndarray) -> bool:
    """True iff dtype, shape and raw bytes are identical (stricter than ``np.array_equal``)."""
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def write_cache(arrays: dict[str, np.ndarray], path: Path, source_sha256: dict[str, str]) -> None:
    """Write the compressed cache atomically and record the digests of its source files."""
    meta = {"format": "ssrk-uci-har-cache/1",
            "note": "0-based labels (file label - 1); X parsed with np.loadtxt(dtype=float32)",
            "source_sha256": source_sha256}
    part = path.with_name(path.name + ".part")
    with open(part, "wb") as fh:
        np.savez_compressed(fh, meta_json=np.array(json.dumps(meta, sort_keys=True)), **arrays)
    os.replace(part, path)


def verify_cache(path: Path, arrays: dict[str, np.ndarray],
                 source_sha256: Optional[dict[str, str]] = None) -> list[str]:
    """Compare the cache at ``path`` with freshly parsed arrays; return the mismatching keys."""
    bad = []
    with np.load(path, allow_pickle=False) as z:
        for key, ref in arrays.items():
            if key not in z.files or not bitwise_equal(z[key], ref):
                bad.append(key)
        if source_sha256 is not None:
            meta = json.loads(str(z["meta_json"])) if "meta_json" in z.files else {}
            if meta.get("source_sha256") != source_sha256:
                bad.append("meta_json.source_sha256")
    return bad


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--zip", type=Path, default=None,
                        help="manually downloaded UCI HAR archive (outer or inner zip)")
    parser.add_argument("--data-dir", type=Path, default=dl.DATA_DIR,
                        help="data root (default: the package's data/ directory)")
    parser.add_argument("--work-dir", type=Path, default=None,
                        help="where the inner archive is unpacked (default: <data-dir>/_downloads)")
    parser.add_argument("--cache", action="store_true",
                        help=f"also write <data-dir>/uci_har/{CACHE_NAME} (verified bitwise)")
    parser.add_argument("--verify-cache", action="store_true",
                        help="check an existing cache against the text files")
    parser.add_argument("--uci-full", action="store_true",
                        help="also extract/verify the raw 'Inertial Signals' (not used)")
    parser.add_argument("--force", action="store_true", help="re-extract even verified files")
    args = parser.parse_args(argv)

    data_dir = args.data_dir.resolve()
    work_dir = (args.work_dir or data_dir / "_downloads").resolve()
    manifest = dl.load_checksums()
    try:
        if args.zip is not None:
            print(f"extracting {args.zip} -> {dl._shown(data_dir / DATASET_SUBDIR)}")
            dl.extract_uci_har(args.zip.resolve(), data_dir, manifest, work_dir,
                               include_optional=args.uci_full, force=args.force)
        print("verifying UCI HAR files against data/checksums.json")
        failures = dl.verify_dataset(data_dir, manifest, "uci_har", args.uci_full)
        if failures:
            print("verification FAILED. Obtain the dataset with `python data/download.py --uci-har` "
                  "or pass the downloaded archive with --zip.", file=sys.stderr)
            return 1
        if not (args.cache or args.verify_cache):
            print("UCI HAR verified; the experiments read the text files directly.")
            return 0

        t0 = time.perf_counter()
        arrays = load_uci_har_text(data_dir / DATASET_SUBDIR)
        check_shapes(arrays)
        print(f"parsed text files in {time.perf_counter() - t0:.1f} s")
        sources = {rel: spec["sha256"]
                   for rel, spec in dl.dataset_files(manifest, "uci_har").items()}
        cache = data_dir / "uci_har" / CACHE_NAME
        if args.cache:
            write_cache(arrays, cache, sources)
            print(f"wrote {dl._shown(cache)} ({cache.stat().st_size / 1e6:.1f} MB)")
        if not cache.is_file():
            raise dl.DataError(f"no cache at {cache}; create it with --cache")
        bad = verify_cache(cache, arrays, sources)
        if bad:
            raise dl.DataError(f"cache {cache} differs from the text files in: {', '.join(bad)}")
        print("cache verified: every array is bitwise identical to np.loadtxt of the text files")
        return 0
    except dl.DataError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
