"""Unit tests of the paired statistics (experiments/paired/paired_stats.py, make_table.py).

* Holm adjustment against a brute-force step-down definition (exact equality), caps, ties, the minimum
  attainable value 5/100001 quoted in the table caption.
* Sign-randomization p value: formula (b + 1) / (10^5 + 1) with b recomputed independently (single-shot draw of
  the same Generator stream, integer arithmetic), degenerate cases, the 10,000-row chunking.
* BCa: degenerate constant input and equality with a direct scipy call.
* Decisions, number formatting of the longtable, row labels, shipped records, the seed guard of the SSRK unit
  loader.
* In-text claims read off Table A3 (Abstract, Sec. 5.1, 5.3, 5.4, 5.5, App. A.3.2 "Baselines"): all pass on the
  original records, and a flipped decision or a changed count makes the corresponding claim fail.
* End-to-end (skipped when the reference SSRK records are not present): the 70 contrasts equal the original
  ``paired_v3_final.csv`` field by field, the longtable body equals the paper's and all 16 claims pass.
"""
from __future__ import annotations

import csv
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pytest

PKG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG))

from experiments.paired import make_table as T  # noqa: E402
from experiments.paired import paired_stats as P  # noqa: E402


# ----------------------------------------------------------------------------------------------- Holm
def holm_bruteforce(p):
    """Holm (1979): sort ascending, adj_(i) = max_{j<=i} min(1, (m - j) p_(j)) (0-based j), mapped back."""
    m = len(p)
    order = sorted(range(m), key=lambda i: (p[i], i))
    adj = [0.0] * m
    for i, idx in enumerate(order):
        adj[idx] = max(min(1.0, (m - j) * p[order[j]]) for j in range(i + 1))
    return adj


def test_holm_known_example():
    p = [0.01, 0.04, 0.03, 0.005, 0.2]
    adj = P.holm_adjust(p)
    assert adj == pytest.approx([0.04, 0.09, 0.09, 0.025, 0.2], abs=1e-15)
    assert adj == holm_bruteforce(p)


@pytest.mark.parametrize("seed", range(20))
def test_holm_random_matches_bruteforce(seed):
    rng = np.random.default_rng(seed)
    m = int(rng.integers(1, 9))
    p = list(rng.uniform(0, 0.6, size=m) ** 2)
    if seed % 3 == 0:  # ties
        p[-1] = p[0]
    adj = P.holm_adjust(p)
    assert adj == holm_bruteforce(p)
    assert all(a >= b for a, b in zip(adj, p))
    assert all(a <= 1.0 for a in adj)
    order = np.argsort(p, kind="stable")
    assert all(adj[order[i]] <= adj[order[i + 1]] for i in range(m - 1))


def test_holm_cap_and_single():
    assert P.holm_adjust([0.5, 0.6, 0.9]) == [1.0, 1.0, 1.0]
    assert P.holm_adjust([0.3]) == [0.3]


def test_holm_minimum_attainable_value():
    pmin = 1 / (P.N_SIGN_FLIPS + 1)
    adj = P.holm_adjust([pmin] * 5)
    assert adj == [5 * pmin] * 5
    assert repr(adj[0]) == "4.9999500004999955e-05"  # as printed in paired_v3_final.csv
    assert T.pfmt(adj[0]) == r"$<\!10^{-4}$"


# ----------------------------------------------------------------------------------------------- sign test
def sign_p_independent(arr, seed, n_draws):
    """Independent re-computation: one (n_draws, n) draw of the same stream; integer statistic n*|mean|."""
    arr = np.asarray(arr, dtype=np.float64)
    assert np.all(arr == np.round(arr)), "integer-valued input keeps the arithmetic exact"
    signs = np.random.default_rng(seed).choice(np.array([-1.0, 1.0]), size=(n_draws, arr.size))
    stat = np.abs(signs.astype(np.int64) @ arr.astype(np.int64))
    observed = abs(int(arr.sum()))
    b = int(np.sum(stat >= observed))
    return b, (b + 1) / (n_draws + 1)


@pytest.mark.parametrize("arr,seed", [
    ([3, -1, 2, 5, 0, 4, -2, 1, 1, 6, 2, -3, 4, 1, 2, 0, 3, 1, -1, 2], 81_000),
    ([1, 1, 1, 1, -1, 1, 1, 1, -1, 1, 1, 1], 81_101),
    ([2, -2, 1, -1, 3, -3, 0, 1], 81_302),
])
def test_sign_randomization_formula(arr, seed):
    b, expected = sign_p_independent(arr, seed, P.N_SIGN_FLIPS)
    p = P.paired_randomization_p(arr, seed)
    assert p == expected == (b + 1) / (100_000 + 1)
    assert 1 / 100_001 <= p <= 1.0


def test_sign_randomization_chunking_and_small_n_draws():
    arr = [2, -1, 3, 1, 0, 2, 1]
    for n_draws in (1, 7, 9_999, 10_000, 10_001, 25_000):
        b, expected = sign_p_independent(arr, 1234, n_draws)
        assert P.paired_randomization_p(arr, 1234, n_draws=n_draws) == expected


def test_sign_randomization_degenerate():
    assert P.paired_randomization_p([0.0] * 20, 1) == 1.0
    # every sign vector attains |mean| = observed  ->  b = 10^5  ->  p = 1 exactly
    assert P.paired_randomization_p([1.0] + [0.0] * 19, 2) == 1.0
    # a constant difference: only the two constant sign vectors are as extreme (prob 2^-19)
    b, expected = sign_p_independent([1] * 20, 3, P.N_SIGN_FLIPS)
    assert P.paired_randomization_p([1.0] * 20, 3) == expected and b <= 3


# ----------------------------------------------------------------------------------------------- BCa
def test_bca_degenerate_constant():
    assert P.bca_ci([0.0] * 20, 71_000) == (0.0, 0.0)
    assert P.bca_ci([-0.5] * 20, 71_000) == (-0.5, -0.5)


def test_bca_matches_scipy_and_is_deterministic():
    from scipy.stats import bootstrap

    x = np.random.default_rng(0).normal(0.1, 0.05, size=20)
    lo, hi = P.bca_ci(x, 71_203)
    ref = bootstrap((x,), np.mean, confidence_level=0.95, n_resamples=10_000, method="BCa",
                    random_state=np.random.default_rng(71_203)).confidence_interval
    assert (lo, hi) == (float(ref.low), float(ref.high))
    assert P.bca_ci(x, 71_203) == (lo, hi)
    assert lo < float(np.mean(x)) < hi


# ----------------------------------------------------------------------------------------------- decisions / format
def test_decisions():
    assert P.decide(0.1, 0.01, "ARI") == "SSRK"
    assert P.decide(-0.1, 0.01, "ARI") == "baseline"
    assert P.decide(-0.1, 0.01, "HNI") == "SSRK"
    assert P.decide(0.1, 0.01, "DomainBalance") == "baseline"
    assert P.decide(0.1, 0.05, "ARI") == "ns"
    assert P.decide(0.1, float("nan"), "ARI") == "ns"


def test_number_formatting():
    assert T.num(0.35486954134334037, 4) == ".3549"
    assert T.num(-0.009304289805146559, 4, signed=True) == "$-$.0093"
    assert T.num(1.95e-5, 4, signed=True) == r"$+2\!\times\!10^{-5}$"
    assert T.num(-4.6e-5, 4, signed=True) == r"$-5\!\times\!10^{-5}$"
    assert T.num(0.0, 4, signed=True) == ".0000"
    assert T.num(13.55, 2, signed=True) == "$+$13.55"
    assert T.num(2.7999999999999994, 3) == "2.800"
    assert T.pfmt(0.00011999880001199988) == r"$1.2\!\times\!10^{-4}$"
    assert T.pfmt(0.0051399486005139945) == ".0051"
    assert T.pfmt(0.5497045029549704) == ".550"
    assert T.pfmt(0.99999) == "1"


# ----------------------------------------------------------------------------------------------- records / guard
def test_shipped_records_complete():
    for dataset, metrics, csv_name, unit_key in P.FAMILIES:
        by_method = P.load_records(csv_name, dataset, unit_key)
        methods = P.METHODS if dataset in {"mnist", "fashion_mnist"} else P.BASELINES
        for m in methods:
            assert len(by_method[m]) == P.N_UNITS, (dataset, m)
            for row in by_method[m].values():
                for met in metrics:
                    v = row[met]
                    assert repr(float(v)) == v or v == str(int(float(v))), (dataset, m, met, v)
        if dataset in {"pbmc", "uci_har"}:
            assert not by_method["SSRK"]  # SSRK of PBMC / UCI HAR is recomputed by this package


def _write_unit(folder: Path, unit: int, seed: int, metrics: dict) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{unit:03d}.json").write_text(json.dumps({"unit": unit, "seed": seed, **metrics}))


def test_seed_guard_rejects_wrong_seed(tmp_path):
    good = {"ARI": 0.3, "NMI": 0.5, "MarkerHits": 14, "CoverageScore": 2.8}
    _write_unit(tmp_path / "ok", 0, P.PBMC_BOOTSTRAP_SEEDS[0], good)
    units, _ = P.load_ssrk_units(tmp_path / "ok", "pbmc", ["ARI", "NMI", "MarkerHits", "CoverageScore"], {})
    assert units == {0: good}
    _write_unit(tmp_path / "bad", 1, 61_000, good)  # e.g. a cached tuning record
    with pytest.raises(RuntimeError, match="is not the evaluation seed"):
        P.load_ssrk_units(tmp_path / "bad", "pbmc", ["ARI"], {})
    uci = {"Accuracy": 0.9, "F1_macro": 0.9, "HNI": 0.0, "DomainBalance": 0.1}
    _write_unit(tmp_path / "uci", 2, P.REPEAT_SEEDS[2], uci)
    eval5 = {"002_SSRK": {"unit": 2, "method": "SSRK", "Accuracy": 0.95, "F1_macro": 0.94}}
    units, _ = P.load_ssrk_units(tmp_path / "uci", "uci_har", list(uci), eval5)
    assert units == {P.REPEAT_SEEDS[2]: {**uci, "Accuracy": 0.95, "F1_macro": 0.94}}
    with pytest.raises(FileNotFoundError):
        P.load_ssrk_units(tmp_path / "missing", "pbmc", ["ARI"], {})


def test_missing_units_fail_loudly(tmp_path):
    _write_unit(tmp_path / "pbmc", 0, P.PBMC_BOOTSTRAP_SEEDS[0],
                {"ARI": 0.3, "NMI": 0.5, "MarkerHits": 14, "CoverageScore": 2.8})
    with pytest.raises(RuntimeError, match="20 required"):
        P.compute({"pbmc": tmp_path / "pbmc", "uci_har": tmp_path / "uci"}, [])


# ----------------------------------------------------------------------------------------------- end to end
REF_SOURCES = (PKG / "reference" / "pbmc" / "units", PKG / "reference" / "uci_har" / "units",
               PKG / "reference" / "uci_har" / "eval5.json")


@pytest.mark.skipif(not all(p.exists() for p in REF_SOURCES), reason="reference SSRK unit records not present")
def test_end_to_end_reference(tmp_path):
    P.main(["--ssrk-source", "reference", "--out", str(tmp_path)])
    with (PKG / "reference" / "paired" / "paired_v3_final.csv").open(encoding="utf-8", newline="") as h:
        ref = list(csv.DictReader(h))
    with (tmp_path / "paired_statistics.csv").open(encoding="utf-8", newline="") as h:
        new = list(csv.DictReader(h))
    assert ref == new
    assert json.loads((tmp_path / "paired_means.json").read_text()) == \
        json.loads((PKG / "reference" / "paired" / "paired_v3_final_means.json").read_text())
    assert json.loads((tmp_path / "decision_counts.json").read_text()) == \
        {"contrasts": 70, "SSRK": 52, "baseline": 4, "ns": 14}
    assert T.main(["--csv", str(tmp_path / "paired_statistics.csv")]) == 0
    claims = json.loads((tmp_path / "intext_claims.json").read_text(encoding="utf-8"))
    assert len(claims) == 16 and all(c["ok"] and c["in_paper"] for c in claims)


def test_table_from_reference_csv(tmp_path):
    """The longtable body typeset from the original paired_v3_final.csv equals the paper's body."""
    idx = T.read_rows(PKG / "reference" / "paired" / "paired_v3_final.csv")
    body = T.longtable_body(idx)
    expected = (PKG / "reference" / "paper" / "paired_longtable_body.tex").read_bytes().decode("utf-8")
    assert body == expected.replace("\r\n", "\n")
    assert sum(1 for _ in itertools.filterfalse(lambda ln: ln.startswith(r"\addlinespace"), body.splitlines())) == 70


def test_row_labels():
    assert set(T.LABELS) == {"paper", "original"}
    idx = T.read_rows(PKG / "reference" / "paired" / "paired_v3_final.csv")
    paper = T.longtable_body(idx).splitlines()
    original = T.longtable_body(idx, "original").splitlines()
    assert paper[0].startswith(r"PBMC 3k ARI$\uparrow$ & Variance & ")
    assert original[0].startswith(r"PBMC3k ARI$\uparrow$ & Variance & ")
    # only the row labels differ
    assert [ln.split(" & ", 1)[1] for ln in paper if " & " in ln] == \
        [ln.split(" & ", 1)[1] for ln in original if " & " in ln]
    with pytest.raises(SystemExit):
        T.main(["--labels", "unknown", "--csv", str(PKG / "reference" / "paired" / "paired_v3_final.csv")])


# ----------------------------------------------------------------------------------------------- in-text claims
REF_CSV = PKG / "reference" / "paired" / "paired_v3_final.csv"
EXCERPTS = PKG / "reference" / "paired" / "paper_intext_excerpts.tex"


def _claims(idx, units_csv=None):
    paras = T.read_excerpts(EXCERPTS)
    claims = {c["id"]: c for c in T.intext_claims(idx, units_csv)}
    for c in claims.values():
        c["in_paper"] = c["text"] in paras[c["paragraph"]]
    return claims


def test_excerpt_file_layout():
    paras = T.read_excerpts(EXCERPTS)
    assert len(paras) == len(T.PARAGRAPHS) == 6
    assert paras[0].endswith("the remaining 14 show no significant difference.")
    assert paras[3].startswith("Table~\\ref{tab:image} reports the MNIST five-run comparison")
    assert paras[4].startswith("All baselines use the selection budgets")
    assert paras[5].startswith("Unsupervised feature selection is widely used")
    assert all(p == p.strip() for p in paras)


def test_intext_claims_hold_on_original_records():
    claims = _claims(T.read_rows(REF_CSV))
    assert {"overview_counts", "abstract_counts", "pbmc_variance_ari", "image_matched_mnist"} <= set(claims)
    assert len(claims) == 15  # + uci_no_appended_column when ssrk_per_unit.csv is available
    assert [k for k, c in claims.items() if not (c["ok"] and c["in_paper"])] == []
    assert claims["pbmc_variance_ari"]["text"].endswith("(ARI 0.048).")
    # Sec. 5.4 refers to MNIST; the same sentence would not describe Fashion-MNIST (5 contrasts differ)
    assert len(claims["image_matched_mnist"]["info"]["fashion_mnist_deviations"]) == 5


@pytest.mark.parametrize("key,decision,claim", [
    (("mnist", "NMI", "Variance"), "SSRK", "image_matched_mnist"),
    (("mnist", "CoverageEff", "Variance"), "ns", "image_matched_mnist"),
    (("mnist", "ARI", "SSFS"), "ns", "image_matched_mnist"),
    (("pbmc", "ARI", "Variance"), "ns", "pbmc_variance_ari"),
])
def test_intext_claim_fails_when_a_decision_flips(key, decision, claim):
    idx = T.read_rows(REF_CSV)
    idx[key] = dict(idx[key], decision=decision)
    assert _claims(idx)[claim]["ok"] is False


def test_count_claims_fail_when_counts_change():
    key = ("fashion_mnist", "NMI", "CAE")  # an n.s. contrast
    idx = T.read_rows(REF_CSV)
    idx[key] = dict(idx[key], decision="SSRK")
    claims = _claims(idx)
    assert not claims["overview_counts"]["in_paper"] and not claims["abstract_counts"]["in_paper"]
    idx = T.read_rows(REF_CSV)
    idx[key] = dict(idx[key], n_pairs="19")
    claims = _claims(idx)
    assert claims["overview_counts"]["ok"] is False and not claims["abstract_counts"]["in_paper"]
