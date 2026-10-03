"""Exact Gaussian fixed-target study: predeclared protocol.

Paper: Sec. 5.2 ("Exact Gaussian fixed-target study") and App. A.4
"Exact Gaussian fixed-target study" (Table 1, Tables A4 / A5 / A6, Figure A1).

Data-generating process (App. A.4, "Data-generating process")
    * candidates ``V_i ~ N(0, Sigma)``, ``Sigma_jk = 0.3^{|j-k|}``, ``p = 80``,
      ``n = 800``;
    * 30 nonnull candidates drawn uniformly without replacement per dataset;
      each enters three distinct targets (uniform over the ``r = 12`` target
      coordinates) with coefficient ``a * sign / sqrt(3)``, random signs;
      target noise ``N(0, 1)``;  ``Y = V B + E``;
    * knockoffs from eq. (3) with known ``Sigma`` and
      ``s = 0.99 min(1, 2 lambda_min(Sigma))`` using a separate generator;
    * the proxy null ``H_0j: Y indep V_j | V_{-j}`` holds exactly for the 50
      zero rows of ``B``; the all-null model sets ``B = 0``.
    The amplitude ``a`` only rescales ``B``: all amplitudes share ``V``, the
    support, target assignments, signs, noise and knockoffs dataset by dataset.

Statistic: mean over ``R`` restarts (predeclared algorithm seeds, one paired
design) of the fixed-target gate statistic ``W = tanh(u/(2 tau))`` trained by
:func:`ssrk.fixed_target.fit_fixed_target_batched`; knockoff+ (eq. (8))
at ``q in {0.05, 0.10, 0.15}`` uses the same ``W``.

Computational protocol (part of what makes the records bitwise reproducible)
    * one *unit* = one dataset with its ``R`` restarts (members);
    * units are packed greedily, in order, into batched calls of at most
      ``MAX_MEMBERS = 640`` members; non-audited units first, audited units in
      their own chunks;
    * every audited chunk is retrained on swapped data with the identical member
      layout (the "retrain on swap_S(D)" operation of Lemma 2).

All functions are faithful ports of the research code; numerics, dtypes and RNG
consumption order are unchanged.  This module imports no torch: the trainer is
imported inside :func:`train_chunk`, so the CPU-only recomputation from the shipped
records (tables, numbers, rescoring) needs only NumPy, SciPy and pandas.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np

from ssrk.ft_config import FixedTargetConfig
from ssrk.gaussian_knockoffs import exact_gaussian_knockoffs
from ssrk.knockoff_plus import knockoff_plus

# ----------------------------------------------------------------------------
# Predeclared constants
# ----------------------------------------------------------------------------

PROTOCOL = {
    "n": 800,
    "p": 80,
    "target_dim": 12,
    "support_size": 30,
    "targets_per_signal": 3,
    "signal": 0.75,
    "noise_sd": 1.0,
    "rho": 0.3,
    "q_levels": [0.05, 0.10, 0.15],
    "signal_seeds": list(range(0, 50)),
    "all_null_seeds": list(range(1000, 1100)),
    "tuning_seeds": list(range(5000, 5020)),
    "tuning_null_seeds": list(range(6000, 6020)),
    "knockoff_seed_offset": 10_000_019,
    "knockoff_s_rule": "s = 0.99 * min(1, 2 * lambda_min(Sigma)), equicorrelated",
}

TUNING_SIGNAL_SEEDS = list(range(5000, 5020))   # disjoint from every evaluation seed
TUNING_NULL_SEEDS = list(range(6000, 6020))
EVAL_ALGORITHM_BASE = 424_242
TUNE_ALGORITHM_BASE = 7_000
ALGORITHM_SEED_STRIDE = 104_729
MAX_MEMBERS = 640
EVAL_SIGNAL_AUDIT_EVERY = 5     # every fifth signal dataset is swap-checked
EVAL_NULL_AUDIT_EVERY = 10      # every tenth all-null dataset is swap-checked
AUDIT_SUBSET_SEED_OFFSET = 900_000

# 100-trajectory paired audit (App. A.4, "Swap checks and compute")
TRAJECTORY_SEED_BASE = 777_000
TRAJECTORY_ALGORITHM_BASE = 13_000

LOCKED_CONFIG_KEY = "T04_R20_lr2e-2_e150"
LOCKED_OVERRIDES: dict = {}
LOCKED_RESTARTS = 20

# The two evaluation invocations of the paper (same seeds, same locked config).
EVAL_MAIN = {"tag": "eval_main", "amplitudes": [0.3, 0.4, 0.5, 0.75], "with_null": True}
EVAL_LOW = {"tag": "eval_low", "amplitudes": [0.15, 0.2], "with_null": False}
EVAL_ALL_TAG = "eval_all"

# Component ablation on the evaluation seeds (Table A6).
ABLATION_AMPLITUDES = [0.2, 0.3]
ABLATIONS = {
    # name: (config_key, overrides, restarts)
    "R1": ("ABL_R1", {}, 1),
    "R5": ("ABL_R5", {}, 5),
    "nosharpen": ("ABL_nosharpen", {"lambda_stage2": 0.0}, 20),
    "noentropy": ("ABL_noentropy", {"lambda_stage1": 0.0, "lambda_stage2": 0.0}, 20),
    "noctx": ("ABL_noctx", {"use_target_context": False, "target_mask_prob": 1.0}, 20),
}

# Predeclared tuning grid (Table A4): key -> (config overrides, restarts).
TUNING_GRID = {
    "T01_R1_lr2e-2_e150": ({}, 1),
    "T02_R5_lr2e-2_e150": ({}, 5),
    "T03_R10_lr2e-2_e150": ({}, 10),
    "T04_R20_lr2e-2_e150": ({}, 20),
    "T05_R10_lr5e-3_e150": ({"lr_gate": 5e-3}, 10),
    "T06_R10_lr5e-3_e300": ({"lr_gate": 5e-3, "epochs": 300}, 10),
    "T07_R10_lr2e-2_e150_nosharpen": ({"lambda_stage2": 0.0}, 10),
    "T08_R10_lr2e-2_e150_noctx": ({"use_target_context": False, "target_mask_prob": 1.0}, 10),
}
TUNING_AMPLITUDES = [0.2, 0.3]

SELECTION_RULE = (
    "Predeclared before running: choose the configuration with the largest mean power at q=0.10, "
    "averaged over the tuning amplitudes, on tuning seeds only; among configurations within 0.01 of "
    "the best, choose the one with the smallest cost (restarts x epochs). FDR is not a selection "
    "criterion because it is controlled by construction for every configuration."
)


# ----------------------------------------------------------------------------
# Data generation
# ----------------------------------------------------------------------------

def ar1_covariance(p: int, rho: float) -> np.ndarray:
    """AR(1) correlation matrix ``Sigma_jk = rho^{|j-k|}``."""
    index = np.arange(p)
    return rho ** np.abs(index[:, None] - index[None, :])


def make_dataset(seed: int, null_model: bool, signal: float | None = None, proto: dict = PROTOCOL):
    """Generate one dataset of the exact Gaussian study.

    Args:
        seed: dataset seed (``np.random.default_rng(seed)`` draws ``V``, the
            support, target assignments, signs and noise in this order).
        null_model: all-null model ``B = 0`` (no support draws are made).
        signal: amplitude ``a`` (``||B_j.||_2 = a`` for nonnull ``j``);
            ``None`` uses ``proto["signal"]``.

    Returns:
        ``(V, V_tilde, Y, B, support, s_value)``; knockoffs use the generator
        ``np.random.default_rng(seed + knockoff_seed_offset)``.
    """
    n, p, r = proto["n"], proto["p"], proto["target_dim"]
    amplitude = proto["signal"] if signal is None else float(signal)
    rng = np.random.default_rng(seed)
    Sigma = ar1_covariance(p, proto["rho"])
    V = rng.multivariate_normal(np.zeros(p), Sigma, size=n)
    B = np.zeros((p, r))
    support: list[int] = []
    if not null_model:
        support = sorted(int(j) for j in rng.choice(p, size=proto["support_size"], replace=False))
        k = proto["targets_per_signal"]
        for j in support:
            targets = rng.choice(r, size=k, replace=False)
            signs = rng.choice([-1.0, 1.0], size=k)
            B[j, targets] = amplitude * signs / math.sqrt(k)
    Y = V @ B + proto["noise_sd"] * rng.normal(size=(n, r))
    knockoff_rng = np.random.default_rng(seed + proto["knockoff_seed_offset"])
    V_tilde, s_value = exact_gaussian_knockoffs(V, Sigma, knockoff_rng)
    return V, V_tilde, Y, B, support, s_value


# ----------------------------------------------------------------------------
# Metrics and summaries
# ----------------------------------------------------------------------------

def fdp_power(selected: list[int], support: list[int]) -> tuple[float, float]:
    """False discovery proportion (0 for an empty selection) and power (nan if no nonnulls)."""
    sel, sup = set(selected), set(support)
    fdp = len(sel - sup) / len(sel) if sel else 0.0
    power = (len(sel & sup) / len(sup)) if sup else float("nan")
    return fdp, power


def bootstrap_ci(values, seed, draws=20_000):
    """Mean and percentile-bootstrap 95% interval over datasets (20,000 resamples)."""
    a = np.asarray(values, float)
    if len(a) == 0:
        return (float("nan"),) * 3
    rng = np.random.default_rng(seed)
    means = a[rng.integers(0, len(a), size=(draws, len(a)))].mean(axis=1)
    return float(a.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def summarize(rows: list[dict]) -> list[dict]:
    """Per (model, amplitude, q) cell: means, bootstrap CIs, SDs, empty rates and,
    for the all-null model, the one-sided 95% Clopper-Pearson upper bound on
    P(nonempty selection) = all-null FDR.  Bootstrap seed of cell ``gi`` (index in
    the sorted cell list) is ``31_000 + gi``."""
    out = []
    groups: dict[tuple[str, float, float], list[dict]] = {}
    for row in rows:
        groups.setdefault((row["model"], row["signal"], row["q"]), []).append(row)
    for gi, ((model, signal, q), group) in enumerate(sorted(groups.items())):
        rec = {"model": model, "signal": signal, "q": q, "datasets": len(group)}
        for metric in ["FDP", "power", "selection_size"]:
            vals = [float(g[metric]) for g in group]
            if model == "all_null" and metric == "power":
                continue
            m, lo, hi = bootstrap_ci(vals, 31_000 + gi)
            rec[f"{metric}_mean"], rec[f"{metric}_ci_low"], rec[f"{metric}_ci_high"] = m, lo, hi
            rec[f"{metric}_sd"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        rec["empty_rate"] = float(np.mean([g["empty"] for g in group]))
        nonempty = [float(g["FDP"]) for g in group if not g["empty"]]
        rec["conditional_FDP_nonempty"] = float(np.mean(nonempty)) if nonempty else float("nan")
        rec["nonempty_count"] = len(nonempty)
        if model == "all_null":
            k = len(nonempty)
            N = len(group)
            # Clopper-Pearson one-sided 95% upper bound on P(nonempty) = all-null FDR.
            from scipy.stats import beta
            rec["nonempty_upper95"] = float(beta.ppf(0.95, k + 1, N - k)) if k < N else 1.0
        out.append(rec)
    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    """CSV with the union of keys in order of first appearance (blank for missing)."""
    if not rows:
        return
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, restval="")
        writer.writeheader()
        writer.writerows(rows)


# ----------------------------------------------------------------------------
# Units, packing and batched training
# ----------------------------------------------------------------------------

def algorithm_seeds(base: int, restarts: int) -> list[int]:
    """Predeclared, data-independent restart seeds shared by every dataset."""
    return [base + ALGORITHM_SEED_STRIDE * k for k in range(restarts)]


def audit_subset(seed: int) -> np.ndarray:
    """Swap subset of an audited dataset: ``|S|`` uniform on {1,...,40}, coordinates uniform."""
    rng = np.random.default_rng(AUDIT_SUBSET_SEED_OFFSET + seed)
    size = int(rng.integers(1, PROTOCOL["p"] // 2 + 1))
    return np.sort(rng.choice(PROTOCOL["p"], size=size, replace=False))


def build_units(seeds, null_model, signal, restarts, base, audit_every):
    """One unit = one dataset with its restarts (and a swap subset if audited)."""
    units = []
    for s in seeds:
        V, Vt, Y, B, support, s_value = make_dataset(s, null_model, signal)
        audited = audit_every > 0 and s % audit_every == 0
        units.append({
            "seed": s, "null": null_model, "signal": 0.0 if null_model else float(signal),
            "V": V, "Vt": Vt, "Y": Y, "support": support, "s_value": s_value,
            "algo": algorithm_seeds(base, restarts),
            "audit_subset": audit_subset(s) if audited else None,
        })
    return units


def pack_units(units):
    """Greedy in-order packing into chunks of at most ``MAX_MEMBERS`` members."""
    chunks, current, size = [], [], 0
    for u in units:
        m = len(u["algo"])
        if current and size + m > MAX_MEMBERS:
            chunks.append(current)
            current, size = [], 0
        current.append(u)
        size += m
    if current:
        chunks.append(current)
    return chunks


def chunk_plan(units):
    """The ordered list of ``(chunk, is_audit_chunk)`` used by :func:`run_units`."""
    audited = [u for u in units if u["audit_subset"] is not None]
    plain = [u for u in units if u["audit_subset"] is None]
    return [(c, False) for c in pack_units(plain)] + [(c, True) for c in pack_units(audited)]


def train_chunk(chunk, config, device, swapped):
    """One batched call over all members of a chunk (unit order, then restart order)."""
    from ssrk.fixed_target import fit_fixed_target_batched  # torch is needed only for training

    Vs, Vts, Ys, seeds = [], [], [], []
    for u in chunk:
        V, Vt = u["V"], u["Vt"]
        if swapped:
            S = u["audit_subset"]
            V, Vt = V.copy(), Vt.copy()
            V[:, S], Vt[:, S] = u["Vt"][:, S], u["V"][:, S]
        for k in range(len(u["algo"])):
            Vs.append(V)
            Vts.append(Vt)
            Ys.append(u["Y"])
            seeds.append(u["algo"][k])
    return fit_fixed_target_batched(np.stack(Vs), np.stack(Vts), np.stack(Ys), config, seeds, device)


def run_units(units, config: FixedTargetConfig, device: str):
    """Train all units (Table A1, steps 3-7) and attach results to them.

    Audited units are packed into their own chunks; each such chunk is retrained
    on the swapped data with an identical member layout, which is exactly the
    "retrain on swap_S(D)" operation of Lemma 2.  The aggregate is the
    coordinatewise mean over restarts (odd rule, App. A.2, paragraph "Aggregation").

    Returns ``(layout, gpu_seconds)``.
    """
    layout, seconds = [], 0.0
    for ci, (chunk, is_audit) in enumerate(chunk_plan(units)):
        fit = train_chunk(chunk, config, device, swapped=False)
        seconds += fit.seconds
        fit_s = None
        if is_audit:
            fit_s = train_chunk(chunk, config, device, swapped=True)
            seconds += fit_s.seconds
        layout.append({"chunk": ci, "audit_chunk": is_audit, "members": int(fit.W.shape[0]),
                       "units": [u["seed"] for u in chunk], "seconds": fit.seconds})
        pos = 0
        for u in chunk:
            R = len(u["algo"])
            idx = list(range(pos, pos + R))
            pos += R
            orig = fit.W[idx]
            u["W_runs"] = orig
            u["W"] = orig.mean(axis=0)
            u["final_loss"] = float(fit.final_loss[idx].mean())
            u["max_frobenius"] = fit.max_frobenius[idx].max(axis=0).tolist()
            u["cap_projections"] = int(fit.cap_projections[idx].sum())
            u["logit_clamp_active"] = int(fit.logit_clamp_active[idx].sum())
            if fit_s is not None:
                sw = fit_s.W[idx]
                mask = np.zeros(PROTOCOL["p"], bool)
                mask[u["audit_subset"]] = True
                expected_runs = np.where(mask[None, :], -orig, orig)
                W_sw = sw.mean(axis=0)
                expected = np.where(mask, -u["W"], u["W"])
                u["swap_audit"] = {
                    "subset_size": int(mask.sum()),
                    "per_run_bitwise_mismatches": int(np.sum(sw != expected_runs)),
                    "per_run_max_abs_error": float(np.max(np.abs(sw - expected_runs))),
                    "aggregate_bitwise_mismatches": int(np.sum(W_sw != expected)),
                    "aggregate_max_abs_error": float(np.max(np.abs(W_sw - expected))),
                    "same_selection_all_q": all(
                        knockoff_plus(W_sw, q)[1] == knockoff_plus(expected, q)[1]
                        for q in PROTOCOL["q_levels"]),
                }
    return layout, seconds


def unit_rows(u):
    """One row per target level: knockoff+ threshold, selection, FDP and power."""
    rows = []
    for q in PROTOCOL["q_levels"]:
        T, selected = knockoff_plus(u["W"], q)
        fdp, power = fdp_power(selected, u["support"])
        rows.append({
            "seed": u["seed"], "model": "all_null" if u["null"] else "signal", "signal": u["signal"],
            "q": q, "threshold": T, "selection_size": len(selected),
            "true_positives": len(set(selected) & set(u["support"])),
            "false_positives": len(set(selected) - set(u["support"])),
            "FDP": fdp, "power": power, "empty": int(len(selected) == 0),
            "selected": " ".join(map(str, selected)),
        })
    return rows


def unit_info(u):
    """Per-dataset record (W, seeds, diagnostics, swap-check result)."""
    sig = np.zeros(PROTOCOL["p"], bool)
    sig[u["support"]] = True
    info = {k: u[k] for k in ("seed", "signal", "support", "s_value", "final_loss", "max_frobenius",
                              "cap_projections", "logit_clamp_active")}
    info["model"] = "all_null" if u["null"] else "signal"
    info["W"] = u["W"].tolist()
    info["algorithm_seeds"] = u["algo"]
    info["null_W_absmax"] = float(np.abs(u["W"][~sig]).max())
    info["signal_W_mean"] = float(u["W"][sig].mean()) if sig.any() else float("nan")
    if "swap_audit" in u:
        info["swap_audit"] = u["swap_audit"]
        info["swap_audit"]["subset"] = u["audit_subset"].tolist()
    return info


# ----------------------------------------------------------------------------
# Tuning selection rule
# ----------------------------------------------------------------------------

def tuning_cost(key: str) -> int:
    """Cost of a grid entry: restarts x epochs."""
    override, R = TUNING_GRID[key]
    return R * int(override.get("epochs", FixedTargetConfig().epochs))


def select_configuration(summary: list[dict], keys: list[str] | None = None) -> dict:
    """Apply the predeclared selection rule (``SELECTION_RULE``) to a tuning summary.

    Score of a configuration = mean over the tuning amplitudes (in increasing
    order) of its mean power at q = 0.10; among configurations within 0.01 of the
    best score, the cheapest (restarts x epochs) is chosen, ties broken by grid
    order.
    """
    keys = [k for k in TUNING_GRID if keys is None or k in keys]
    scores, cost = {}, {}
    for key in keys:
        cells = sorted((r for r in summary if r["config_key"] == key and r["model"] == "signal"
                        and abs(r["q"] - 0.10) < 1e-9), key=lambda r: r["tuning_amplitude"])
        scores[key] = float(np.mean([r["power_mean"] for r in cells]))
        cost[key] = tuning_cost(key)
    best = max(scores.values())
    eligible = [k for k in keys if scores[k] >= best - 0.01]
    chosen = min(eligible, key=lambda k: cost[k])
    return {"scores": scores, "cost": cost, "chosen": chosen,
            "override": TUNING_GRID[chosen][0], "restarts": TUNING_GRID[chosen][1]}


# ----------------------------------------------------------------------------
# 100-trajectory paired audit
# ----------------------------------------------------------------------------

def trajectory_inputs(k: int):
    """Dataset (fresh seed, a = 0.75) and swap subset (|S| uniform on {1,...,80}) of trajectory k."""
    rng = np.random.default_rng(TRAJECTORY_SEED_BASE + k)
    dseed = int(rng.integers(0, 2**31 - 1))
    V, Vt, Y, _, _, _ = make_dataset(dseed, null_model=False)
    size = int(rng.integers(1, PROTOCOL["p"] + 1))
    S = np.sort(rng.choice(PROTOCOL["p"], size=size, replace=False))
    return V, Vt, Y, S
