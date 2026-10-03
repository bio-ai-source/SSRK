"""Deterministic-execution settings and global seeding.

Three entry points are shared by the whole package:

``enforce_cublas_workspace()``
    Sets ``CUBLAS_WORKSPACE_CONFIG=:4096:8``, the value used for every record of the paper.  A
    different pre-set value (e.g. ``:16:8``, the other value allowed by deterministic cuBLAS) is
    overridden with a warning on stderr: the workspace size can change cuBLAS algorithm selection and
    hence the low-order bits of W.  The variable is read when the first cuBLAS handle is created, so
    the experiment scripts call this before importing torch.  This function imports no torch.

``configure_determinism()``
    Settings used by the fixed-target (exact Gaussian) study: deterministic
    algorithms, cuDNN deterministic / no benchmark, TF32 disabled for matmul and
    cuDNN, and ``CUBLAS_WORKSPACE_CONFIG=:4096:8`` (via ``enforce_cublas_workspace``;
    effective only before the first cuBLAS call, which is why the experiment
    scripts also set it before importing torch).  Identical to the settings that
    produced the paper's records of the exact Gaussian study.

``set_seed(seed, deterministic=True)``
    Global seeding used by the self-reconstruction (ranking) mode: Python
    ``random``, NumPy's legacy global generator, the torch CPU generator and all
    CUDA generators, plus the deterministic flags.  The body is a verbatim copy of
    the research code's ``ssrk.training.set_seed`` so that the RNG streams (and
    hence every downstream draw) are identical.

None of the functions has import-time side effects, and importing this module does not import
torch (the functions that need it import it when called).
"""

from __future__ import annotations

import os
import random
import sys

import numpy as np

CUBLAS_WORKSPACE = ":4096:8"


def enforce_cublas_workspace() -> str | None:
    """Set ``CUBLAS_WORKSPACE_CONFIG`` to ``:4096:8``; warn if another value was set.

    Returns the previous value (``None`` if the variable was not set).
    """
    previous = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if previous is not None and previous != CUBLAS_WORKSPACE:
        print(f"[ssrk] warning: CUBLAS_WORKSPACE_CONFIG={previous!r} overridden with {CUBLAS_WORKSPACE!r} "
              "(the value used for the paper's records; another workspace size can change cuBLAS kernels "
              "and the low-order bits of W)", file=sys.stderr, flush=True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = CUBLAS_WORKSPACE
    return previous


def configure_determinism() -> None:
    """Deterministic kernels for the batched fixed-target trainer (no TF32)."""
    import torch

    enforce_cublas_workspace()
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed ``random``, ``np.random`` (legacy), torch CPU and CUDA generators.

    Args:
        seed: random seed value.
        deterministic: if True, also request deterministic cuDNN/cuBLAS kernels
            (same flags, same order as the research code).
    """
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = CUBLAS_WORKSPACE
        if hasattr(torch, "use_deterministic_algorithms"):
            try:
                torch.use_deterministic_algorithms(True)
            except Exception as exc:  # pragma: no cover
                print(f"[ssrk] warning: torch.use_deterministic_algorithms(True) failed ({exc}); "
                      "W may not be bitwise reproducible", file=sys.stderr, flush=True)
