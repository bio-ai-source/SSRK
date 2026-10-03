# Reference environment

All numbers in the paper that this package reproduces were produced on the machine below. The package was also
run end to end on this machine (`run_all.sh`), from a clean copy and a freshly created virtual environment; the
report and the log of that run are in `reference/verification_record/`.

| Component | Reference value |
|---|---|
| GPU | NVIDIA GeForce RTX 4090 Laptop GPU, 16 GB, compute capability 8.9, 76 SMs |
| NVIDIA driver | 595.71 |
| CUDA runtime / cuDNN | 12.1 / 9.1.0 (bundled with the `torch==2.5.1+cu121` wheel) |
| CPU | 13th Gen Intel Core i9-13900H, 20 logical cores (OpenBLAS kernel family reported by `threadpoolctl`: `Haswell`) |
| OS | Windows 11 (10.0.26200), x64 |
| Python | 3.12.12 (CPython, installed with `uv`) |
| Key packages | torch 2.5.1+cu121, numpy 2.5.3 (OpenBLAS 0.3.34), scipy 1.18.1, pandas 3.0.6, scikit-learn 1.9.1, anndata 0.13.4, h5py 3.16.0, threadpoolctl 3.7.0, matplotlib 3.11.2 |

## Requirement files

| File | Use |
|---|---|
| `requirements.txt` | direct dependencies, pinned; everything (training, experiments, verification) |
| `requirements-lock.txt` | full transitive closure of `requirements.txt`, pinned to the versions of the reference environment (Windows x64) |
| `requirements-levelA.txt` | the same pins without torch: enough for `verify.py --from-reference`, also on machines without a CUDA build of torch |

`torch==2.5.1+cu121` exists for Windows x64 and Linux x86-64 only. On Linux, pip also installs the
`nvidia-*-cu12` runtime wheels that this build depends on. Disk space: the environment takes about 4.7 GB
(mostly torch); the package, the data and all results take under 0.5 GB.

## Creating the environment

With `uv` (what we used):

```bash
uv venv --python 3.12.12 .venv
uv pip install --python .venv -r requirements.txt --index-strategy unsafe-best-match
```

With plain `pip` (Python 3.12):

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt          # Windows: .venv\Scripts\python -m pip install -r requirements.txt
```

For `verify.py --from-reference` alone, `requirements-levelA.txt` can replace `requirements.txt`.

## Determinism settings used by the code

The experiment scripts set these themselves; they are listed here so that they can be checked.

* `CUBLAS_WORKSPACE_CONFIG=:4096:8` is set before CUDA is initialised; a different pre-set value is overridden with a
  warning. `torch.use_deterministic_algorithms(True)`, `torch.backends.cudnn.deterministic=True` and
  `torch.backends.cudnn.benchmark=False` are always set; TF32 is disabled for the fixed-target trainer (the
  self-reconstruction trainer keeps the torch defaults, as in the original runs).
* Every random stream is seeded explicitly: data generation, knockoff and reference draws, network initialisation,
  minibatch order, masks, bootstrap resamples and evaluation seeds. The seeds are written to each run's `manifest.json`.
* Scikit-learn evaluation runs with one BLAS/OpenMP thread (`threadpoolctl`). This matters for the UCI HAR MLP,
  whose accuracy depends on the BLAS thread count. The PBMC clustering metrics of SSRK were checked to be
  identical with 1, 2, 4 and 20 threads.

## What "exactly" means on other hardware

On the reference stack, retraining reproduces every statistic bit for bit (`verify.py` checks the raw arrays).
The code fixes every seed, every operation order and every determinism switch. What can still change on another
GPU model, driver or library build is the choice of floating-point kernels, for example cuBLAS GEMM algorithms and
their reduction order. Such a change moves the learned statistics in their low-order bits, and training can
amplify it. CUDA random streams (masks, minibatch permutations) can also depend on the GPU model, because the
launch configuration of PyTorch's random kernels depends on the number of multiprocessors. This is why we report the
reference GPU exactly.

One check on the reference GPU bounds the kernel effect for the fixed-target trainer. That trainer trains many
independent networks in batched calls. Repacking the paper evaluation into calls of 320 instead of 640 members
changes the shapes of all batched matrix products. Every statistic, every selection and Tables 1 and A5 stayed
bitwise unchanged. `run.py` keeps the original packing anyway.

If your raw arrays differ from `reference/`, every reported number can still be recomputed exactly from the
shipped per-run statistics (`python verify.py --from-reference`). The raw comparisons in
`verify.py` show where your run differs.

For CPU-side numerics (data generation with NumPy/LAPACK, PCA and k-means evaluation, the scikit-learn MLP), the
reference machine used OpenBLAS's `Haswell` kernels. On another x86-64 CPU with AVX2, you can force the same
kernels by setting `OPENBLAS_CORETYPE=Haswell` before starting Python.
