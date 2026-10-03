"""UCI HAR data for the matched protocol: loader, feature names/domains, appended permuted columns.

The official split of the UCI HAR dataset (7,352 training / 2,947 test windows x 561 engineered
features; activity labels 1..6 stored 0-indexed; paper Table A2 and App. A.7). SSRK is fitted on the
training split augmented with 561 *appended permuted diagnostic columns* (at ratio 1.0, one
within-column permutation of each real feature, in random column order). They enter the SSRK fit, but
are excluded from the evaluated ranking and serve only as known nulls for the HNI diagnostic; the
ranking that is evaluated is the top 384 of the 561 real features.

The loader returns exactly the arrays of the original ``load_uci_har`` (``np.loadtxt`` float32 / int)
and fails loudly if a file is missing. The feature names (needed for the domain-imbalance metric) can
also be read from the byte copy of ``features.txt`` shipped in ``reference/uci_har/`` (CC BY 4.0, see
``reference/uci_har/README_features.md``), so that the CPU rescoring of Level A needs no download.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Sequence, Set, Tuple

import numpy as np

UCI_SUBDIR = Path("uci_har") / "UCI HAR Dataset"
UCI_TRAIN_SHAPE = (7352, 561)
UCI_TEST_SHAPE = (2947, 561)
#: Byte copy of the dataset's ``features.txt`` shipped with the package (used when the data are absent).
SHIPPED_FEATURES = Path(__file__).resolve().parents[2] / "reference" / "uci_har" / "features.txt"


def uci_root(data_dir: Path) -> Path:
    return Path(data_dir) / UCI_SUBDIR


def _require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(
            f"UCI HAR file not found: {path}. Run data/download.py (or see data/README.md) to fetch and "
            "unpack the official archive."
        )
    return path


def load_uci_har(data_dir: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Official split: ``X_train`` (7352, 561) float32, ``y_train`` int (0..5), ``X_test`` (2947, 561), ``y_test``."""
    root = uci_root(data_dir)
    X_train = np.loadtxt(_require(root / "train" / "X_train.txt"), dtype=np.float32)
    y_train = np.loadtxt(_require(root / "train" / "y_train.txt"), dtype=int) - 1  # 0-indexed
    X_test = np.loadtxt(_require(root / "test" / "X_test.txt"), dtype=np.float32)
    y_test = np.loadtxt(_require(root / "test" / "y_test.txt"), dtype=int) - 1
    if X_train.shape != UCI_TRAIN_SHAPE or X_test.shape != UCI_TEST_SHAPE:
        raise RuntimeError(f"UCI protocol shapes wrong: train={X_train.shape}, test={X_test.shape}")
    return X_train, y_train, X_test, y_test


def features_file(data_dir: Path, allow_shipped_copy: bool = False) -> Path:
    """``features.txt`` of the dataset in ``data_dir``; if it is absent and ``allow_shipped_copy``, the
    copy shipped in ``reference/uci_har/features.txt``."""
    path = uci_root(data_dir) / "features.txt"
    if path.exists() or not allow_shipped_copy:
        return _require(path)
    return _require(SHIPPED_FEATURES)


def load_feature_names(data_dir: Path, allow_shipped_copy: bool = False) -> List[str]:
    """The 561 feature names of ``features.txt`` (second whitespace-separated field)."""
    return read_feature_names(features_file(data_dir, allow_shipped_copy))


def read_feature_names(path: Path) -> List[str]:
    """Parse a ``features.txt`` file (``"<index> <name>"`` per line) into the 561 names."""
    path = _require(Path(path))
    names: List[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split(maxsplit=1)
        if len(parts) == 2:
            names.append(parts[1])
    if len(names) != 561:
        raise RuntimeError(f"Expected 561 UCI feature names, got {len(names)}")
    return names


def uci_domain(feature_name: str) -> str:
    """``"freq"`` for frequency-domain features (names starting with ``f``), else ``"time"``."""
    return "freq" if feature_name.startswith("f") else "time"


def uci_domain_balance(selected: Sequence[int], feature_names: Sequence[str]) -> float:
    """Domain imbalance ``|time_selected/time_total - freq_selected/freq_total|`` (lower is better).

    The group-normalized version of the matched protocol (paper App. A.8: ``|n_time/272 - n_freq/289|``,
    angle features counted as time).
    """
    domains = [uci_domain(feature_names[int(i)]) for i in selected]
    time_count = domains.count("time")
    freq_count = domains.count("freq")
    time_total = sum(uci_domain(name) == "time" for name in feature_names)
    freq_total = sum(uci_domain(name) == "freq" for name in feature_names)
    time_rate = time_count / max(1, time_total)
    freq_rate = freq_count / max(1, freq_total)
    return float(abs(time_rate - freq_rate))


def n_appended(p: int, ratio: float) -> int:
    """Number of appended permuted columns for ``p`` real columns: ``max(1, round(p * ratio))``."""
    return max(1, int(round(p * ratio)))


def appended_columns(p: int, ratio: float) -> Set[int]:
    """Indices of the appended columns in the augmented matrix: ``p, ..., p + n_appended - 1``."""
    return set(range(p, p + n_appended(p, ratio)))


def append_permuted_nulls(X: np.ndarray, ratio: float, seed: int) -> Tuple[np.ndarray, Set[int]]:
    """Append ``round(p * ratio)`` permuted copies of randomly chosen columns (known nulls for HNI).

    ``default_rng(seed)``: source columns ``rng.choice(p, n_null, replace=n_null > p)`` (without
    replacement at ratio 1.0, so every real column is copied exactly once), then one ``rng.permutation``
    per appended column. Returns ``(X_aug float32, set of appended column indices)``.
    """
    n, p = X.shape
    n_null = n_appended(p, ratio)
    rng = np.random.default_rng(seed)
    source_cols = rng.choice(p, size=n_null, replace=n_null > p)
    nulls = np.empty((n, n_null), dtype=np.float32)
    for j, source in enumerate(source_cols):
        nulls[:, j] = rng.permutation(X[:, int(source)])
    return np.hstack([X, nulls]).astype(np.float32), appended_columns(p, ratio)
