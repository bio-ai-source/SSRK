# Self-Supervised Reconstruction Knockoffs (SSRK)

Code for the NeurIPS 2026 paper **Self-Supervised Reconstruction Knockoffs for Calibrated Unsupervised Feature
Selection**.

## Method

Unsupervised feature selection usually scores each feature on an absolute scale: how much it helps some objective.
SSRK instead compares every feature with a matched reference. The feature counts as relevant only if it does better
than its reference on a label-free proxy task, masked reconstruction. The comparison is built into training through a
knockoff-gated network, and the result is a signed statistic `W_j` per feature that can be thresholded with knockoff+.

**Fixed-target mode (controlled discovery).** The features of each observation are split, before looking at the
data, into candidates `V` and a held-out target view `Y`. Knockoffs `Ṽ` are generated for the candidates only. The
predictor never sees a candidate pair directly, only through a gated mixture

```
H_j = π_j V_j + (1 − π_j) Ṽ_j,        π_j = σ(u_j / τ),
```

and it predicts masked entries of `Y` from the masked mixture and the visible entries of `Y`. The reversed ordering
of a pair uses the tied logit `−u_j`, so swapping `V_j` with `Ṽ_j` and negating `u_j` leaves the loss unchanged.
The gates are trained with a two-stage entropy schedule: Stage I pulls `π` toward 1/2, Stage II sharpens it. The
statistic is

```
W_j = 2 π_j − 1 = tanh(u_j / (2τ)).
```

With zero-initialized gate logits and a sign-equivariant optimizer (AdamW here), retraining on data in which any
subset of pairs is swapped gives exactly `−W_j` on that subset (flip-sign). With exact knockoffs, knockoff+ applied
to `W` then controls the false discovery rate of the conditional nulls `Y ⊥ V_j | V_{-j}`. Averaging `W` over restarts
on the same paired design keeps the flip-sign property. The code evaluates the mixture in the algebraically identical
centered form `H_j = (V_j + Ṽ_j)/2 + t_j (V_j − Ṽ_j)/2` with an exactly odd `t_j = tanh(u_j/(2τ))`. In this form
the flip-sign property holds bit for bit in floating point, and the tests check it.

**Self-reconstruction mode (ranking).** Without a separate target view, the masked candidates themselves are the
target. On masked entries, the loss of pair `j` is `π_j ‖X̂_j − X_j‖² + (1 − π_j) ‖X̂_j − X̃_j‖²`. The references are
matched-marginal draws: an independent Gaussian with the feature's mean and variance, or a within-column
permutation. In population, the contrast between a feature and such a reference measures the information the
feature shares with the other features: a feature independent of all others gains nothing, whatever its variance.
The output `W` is a ranking. This mode is used for the real-data experiments (PBMC 3k, UCI HAR).

The package implements both modes (`ssrk/`) and the scripts that reproduce the paper's experiments
(`experiments/`).

## Installation

Python 3.12 and an NVIDIA GPU with CUDA 12.1 support (8 GB of memory are enough):

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt        # Windows: .venv\Scripts\python -m pip install -r requirements.txt
```

`requirements.txt` pins the versions used for the paper, including `torch==2.5.1+cu121`. `ENVIRONMENT.md` describes the
reference machine and the determinism settings. Run the commands below from the package root, with the
environment's Python.

## Using SSRK

**Controlled discovery with a held-out target view.** You need `V` (n × p) with a knockoff sampler, and `Y` (n × r).
In this example the candidates are Gaussian with known covariance, so exact knockoffs are available:

```python
import numpy as np
from ssrk.fixed_target import FixedTargetConfig, fit_fixed_target_batched
from ssrk.gaussian_knockoffs import exact_gaussian_knockoffs
from ssrk.knockoff_plus import knockoff_plus

rng = np.random.default_rng(0)
n, p, r = 800, 80, 12
Sigma = 0.3 ** np.abs(np.subtract.outer(np.arange(p), np.arange(p)))
V = rng.multivariate_normal(np.zeros(p), Sigma, size=n)
B = np.zeros((p, r)); B[:30] = rng.normal(scale=0.3, size=(30, r))
Y = V @ B + rng.normal(size=(n, r))

V_tilde, _ = exact_gaussian_knockoffs(V, Sigma, np.random.default_rng(1))   # uses V only, never Y

R = 20                                                    # restarts on the same paired design, one batched call
seeds = [424242 + 104729 * k for k in range(R)]
fit = fit_fixed_target_batched(np.stack([V] * R), np.stack([V_tilde] * R), np.stack([Y] * R),
                               FixedTargetConfig(), seeds, device="cuda")
W = fit.W.mean(axis=0)
T, selected = knockoff_plus(W, q=0.10)
```

`FixedTargetConfig()` holds the configuration selected in the paper (network widths 256-128-32-128-256, 150 epochs,
gate learning rate 0.02, entropy schedule, logit and norm bounds).

**Ranking without a target view.** Standardize the data, choose a matched-marginal reference
(`"independent"` or `"permutation"`), and rank by `W`:

```python
import numpy as np
from ssrk.selfrecon import SelfReconConfig, resolve_device, ssrk_scores
from ssrk.ranking import deterministic_top_k

X = ...  # (n, p) array, standardized, float32
config = SelfReconConfig(                                 # the setting of the PBMC experiment
    encoder_dims=(256, 128), latent_dim=32, decoder_dims=(128, 256), temperature=1.0, use_batchnorm=True,
    epochs=200, batch_size=128, lr=3e-4, lr_gate_factor=0.05, lambda_entropy=0.005, mask_prob=0.5,
    stage1_frac=0.5)
W, W_restarts, _ = ssrk_scores(X, fit_seed=0, reference="independent", config=config,
                               device=resolve_device("auto"), restarts=1)
top = deterministic_top_k(W, k=100)
```

`experiments/pbmc/run.py` and `experiments/uci_har/run.py` contain the configurations of the two real-data
experiments (`pbmc_config()`, `uci_config()`).

## Data

* **PBMC 3k** is included: `data/pbmc/pbmc3k_processed.h5ad`, 52.6 MB, the processed besca file from Zenodo
  (doi:10.5281/zenodo.3886414). Zenodo distributes this file under AGPL-3.0-only, independently of the code; see
  `data/pbmc/LICENSE` and `data/pbmc/NOTICE.md`.
* **UCI HAR** (282.6 MB extracted) is downloaded from the UCI Machine Learning Repository (CC BY 4.0). The script
  extracts the files the experiments read and checks their SHA-256:

  ```bash
  python data/download.py --uci-har
  ```

  If you downloaded the archive by hand, run `python data/prepare_uci_har.py --zip <archive>` instead.

`data/README.md` lists sources, checksums and preprocessing. The exact Gaussian study generates its own data.

## Reproducing the experiments of the paper

`run_all.sh` (Linux) and `run_all.ps1` (Windows) run everything in order: data, unit tests, all experiments, tables
and a final comparison with the original results. On an RTX 4090 Laptop GPU this takes about 1 h 30 min.

```bash
PYTHON=.venv/bin/python bash run_all.sh
```

```powershell
powershell -ExecutionPolicy Bypass -File .\run_all.ps1 -Python .venv\Scripts\python.exe
```

The individual steps are listed below; outputs go to `results/`.

**Exact Gaussian fixed-target study** (Table 1, Tables A4-A6, Figure A1):

```bash
python experiments/exact_gaussian/run.py main        # evaluation: a = 0.15-0.75, 50 datasets each, 100 all-null datasets   ~13 min
python experiments/exact_gaussian/run.py ablation    # component ablation (Table A6)                                       ~11 min
python experiments/exact_gaussian/run.py tune        # tuning grid on disjoint tuning datasets (Table A4)                  ~10 min
python experiments/exact_gaussian/run.py audit       # 100 paired swap-retraining checks, float32 and float64              ~3 min
python experiments/exact_gaussian/make_tables.py --check    # writes results/exact_gaussian/tables/gaussian_*.tex
python experiments/exact_gaussian/make_figure.py --check    # writes results/exact_gaussian/figures/FigureA1_Gaussian.pdf
```

**PBMC 3k** (Table A3, PBMC rows; Table A8), 20 bootstrap refits:

```bash
python experiments/pbmc/run.py --units all           # ~11 min
python experiments/pbmc/stability.py --strict        # stability table, results/pbmc/stability_body.tex
```

**UCI HAR** (Table A3, UCI HAR rows), 20 training seeds with three restarts each:

```bash
python experiments/uci_har/run.py --units all        # ~31 min
python experiments/uci_har/evaluate.py --workers 4   # MLP accuracy / macro-F1 averaged over five evaluator seeds, ~5 min
```

**Paired comparison with the baselines** (Table A3). The baselines are not part of this package. Their per-run scores,
and the per-run results of the image benchmarks, are included in `experiments/paired/records/`:

```bash
python experiments/paired/paired_stats.py            # results/paired/paired_statistics.csv
python experiments/paired/make_table.py              # results/paired/paired_longtable_body.tex
```

**Comparing with the original results.** `reference/` contains the records of the runs reported in the paper: the
statistics `W` of every dataset and restart, the selections and the per-run metrics. `python verify.py` compares
your `results/` with them bit for bit. It also regenerates the tables and checks them, and the numbers quoted in the
text, against the paper. `python verify.py --from-reference` recomputes all tables and numbers from the shipped
records without retraining. It needs neither a GPU nor torch: `pip install -r requirements-levelA.txt` is enough.
Bitwise agreement of retrained statistics was obtained on the reference machine (`ENVIRONMENT.md`). On other GPUs,
the statistics can differ in their last bits.

```bash
python -m pytest -q tests                            # unit tests, ~30 s
```

The image benchmarks, the representative single runs and the timing measurements of the
paper are not part of this package.

## Repository structure

```
ssrk/            the method: fixed-target mode, self-reconstruction mode, knockoffs, knockoff+
experiments/     exact_gaussian/, pbmc/, uci_har/, paired/: experiment scripts of the paper
data/            PBMC 3k file, UCI HAR download and processing
reference/       records of the original runs and the paper's tables (see reference/README.md)
tests/           unit tests
verify.py        comparison of results/ with reference/ and the paper
```

## Citation

```bibtex
@inproceedings{ssrk2026,
  title     = {Self-Supervised Reconstruction Knockoffs for Calibrated Unsupervised Feature Selection},
  author    = {Huang, Haihui and Kang, Dingkui and Zhou, Yanan and Liang, Yong},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2026}
}
```

License: MIT (see `LICENSE`). The PBMC data file keeps its own license (AGPL-3.0-only).
