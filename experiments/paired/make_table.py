"""Typeset the body of the paper's Table A3 and check it, and the in-text statements read off it, against the paper.

Reads ``results/paired/paired_statistics.csv`` (from ``paired_stats.py``) and writes
``results/paired/paired_longtable_body.tex``: the rows between ``\\endlastfoot`` and ``\\end{longtable}`` of
Table A3, "Matched comparison results for all 70 contrasts" (App. A.3.3 "Matched comparison"; LaTeX label
``tab:paired_summary``).  The body is compared byte-for-byte with the paper's body
(``reference/paper/paired_longtable_body.tex``, a verbatim copy; or extracted from a full ``neurips_2026.tex``
given with ``--paper``).

It also checks the numbers and significance statements of the paper that are read off this table against the
verbatim source lines in ``reference/paired/paper_intext_excerpts.tex`` and writes ``intext_claims.json``:

* Abstract and Sec. 5.1: the counts of the 70 comparisons (52 SSRK / 4 baseline / 14 n.s.);
* Sec. 5.3: PBMC 3k ARI / NMI / marker / coverage statements;
* Sec. 5.4: the matched-protocol image sentence (MNIST, the dataset of that paragraph);
* Sec. 5.5: UCI HAR accuracy / macro-F1 / HNI / domain-imbalance statements;
* App. A.3.2 "Baselines": the PBMC 3k ARI of Variance.

Each claim passes only if its text (formatted from the recomputed contrasts) occurs verbatim in its paragraph
and the statistical condition it asserts holds.  With ``--paper`` every excerpt must also occur verbatim in the
given paper source.

Number formatting (:func:`num`, :func:`pfmt`) is that of the table emitter of the original study.  Row labels follow the final paper (``--labels paper``, default); ``--labels original`` gives
the labels of the original emitter (``PBMC3k``, ``MarkerHits``, ...).

Exit status: 0 only if the table and every in-text claim pass, 1 otherwise.

usage: python experiments/paired/make_table.py [--csv FILE] [--out DIR] [--labels paper|original]
           [--paper neurips_2026.tex | --expected BODY.tex] [--units-csv FILE] [--excerpts FILE]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Sequence


def _rel(path) -> str:
    """``path`` relative to the package root when it lies inside it (for messages)."""
    p = Path(path)
    try:
        return p.resolve().relative_to(Path(__file__).resolve().parents[2]).as_posix()
    except ValueError:
        return str(p)

PKG = Path(__file__).resolve().parents[2]
if str(PKG) not in sys.path:
    sys.path.insert(0, str(PKG))

ORDER_DS = ["pbmc", "mnist", "fashion_mnist", "uci_har"]
ORDER_MET = {
    "pbmc": ["ARI", "NMI", "MarkerHits", "CoverageScore"],
    "mnist": ["ARI", "NMI", "CoverageEff"],
    "fashion_mnist": ["ARI", "NMI", "CoverageEff"],
    "uci_har": ["Accuracy", "F1_macro", "HNI", "DomainBalance"],
}
BASELINES = ["Variance", "STG", "CAE", "GAEFS", "SSFS"]
DECIMALS = {"MarkerHits": 2, "CoverageScore": 3}
DEFAULT_DEC = 4

#: Row labels of the final paper.
PAPER_DS_LABEL = {"pbmc": "PBMC 3k", "mnist": "MNIST", "fashion_mnist": "Fashion-MNIST", "uci_har": "UCI HAR"}
PAPER_MET_LABEL = {
    "ARI": r"ARI$\uparrow$", "NMI": r"NMI$\uparrow$", "MarkerHits": r"marker hits$\uparrow$",
    "CoverageScore": r"coverage score$\uparrow$", "CoverageEff": r"coverage efficiency$\uparrow$",
    "Accuracy": r"Accuracy$\uparrow$", "F1_macro": r"Macro-F1$\uparrow$", "HNI": r"HNI$\downarrow$",
    "DomainBalance": r"domain imbalance$\downarrow$",
}
#: Row labels of the original emitter (make_paired_tables.py).
ORIGINAL_DS_LABEL = {"pbmc": "PBMC3k", "mnist": "MNIST", "fashion_mnist": "Fashion-MNIST", "uci_har": "UCI HAR"}
ORIGINAL_MET_LABEL = {
    "ARI": r"ARI$\uparrow$", "NMI": r"NMI$\uparrow$", "MarkerHits": r"MarkerHits$\uparrow$",
    "CoverageScore": r"CoverageScore$\uparrow$", "CoverageEff": r"CoverageEff$\uparrow$",
    "Accuracy": r"Accuracy$\uparrow$", "F1_macro": r"Macro-F1$\uparrow$", "HNI": r"HNI$\downarrow$",
    "DomainBalance": r"DomainBalance$\downarrow$",
}
LABELS = {"paper": (PAPER_DS_LABEL, PAPER_MET_LABEL), "original": (ORIGINAL_DS_LABEL, ORIGINAL_MET_LABEL)}
#: The four learned baselines (all but Variance), as grouped in Sec. 5.4.
LEARNED = ("STG", "CAE", "GAEFS", "SSFS")
#: Paragraph index of each excerpt line in ``reference/paired/paper_intext_excerpts.tex``.
PARAGRAPHS = {0: "Sec. 5.1", 1: "Sec. 5.3, PBMC 3k", 2: "Sec. 5.5, UCI HAR", 3: "Sec. 5.4, image benchmarks",
              4: 'App. A.3.2, "Baselines"', 5: "Abstract"}


def num(x, dec, signed=False):
    """Fixed-decimal number without leading zero; LaTeX minus; optional explicit plus.

    A signed value that rounds to zero but is not zero is printed in scientific form (e.g. ``$+2\\!\\times\\!10^{-5}$``);
    the sign is taken from the unrounded value.
    """
    x = float(x)
    if signed and 0 < abs(x) < 0.5 * 10 ** (-dec):
        m, e = f"{abs(x):.0e}".split("e")
        return ("$-" if x < 0 else "$+") + rf"{m}\!\times\!10^{{{int(e)}}}$"
    s = f"{abs(x):.{dec}f}"
    if s.startswith("0."):
        s = s[1:]
    if x < 0 and (signed or float(s) != 0):
        return "$-$" + s
    if signed and x > 0:
        return "$+$" + s
    return s


def pfmt(p):
    """Holm-adjusted p value as printed: ``<10^-4``, ``m.m x 10^-e`` below 1e-3, 4 decimals below .01, else 3."""
    if p < 1e-4:
        return r"$<\!10^{-4}$"
    if p < 1e-3:
        m, e = f"{p:.1e}".split("e")
        return rf"${m}\!\times\!10^{{{int(e)}}}$"
    if p < 0.01:
        return f"{p:.4f}"[1:]
    if p >= 0.9995:
        return "1"
    return f"{p:.3f}"[1:]


def read_rows(path: Path) -> dict[tuple[str, str, str], dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as h:
        rows = list(csv.DictReader(h))
    if len(rows) != 70:
        raise RuntimeError(f"{path}: {len(rows)} contrasts, expected 70")
    return {(r["dataset"], r["metric"], r["contrast"].replace("SSRK - ", "")): r for r in rows}


def longtable_body(idx: dict[tuple[str, str, str], dict[str, str]], labels: str = "paper") -> str:
    """Body rows of the longtable (5 rows per dataset--metric family, ``\\addlinespace[2pt]`` between families)."""
    ds_label, met_label = LABELS[labels]
    L = []
    first = True
    for ds in ORDER_DS:
        for met in ORDER_MET[ds]:
            dec = DECIMALS.get(met, DEFAULT_DEC)
            if not first:
                L.append(r"\addlinespace[2pt]")
            first = False
            for i, b in enumerate(BASELINES):
                r = idx[(ds, met, b)]
                lab = f"{ds_label[ds]} {met_label[met]}" if i == 0 else ""
                ss = num(float(r["ssrk_mean"]), dec) if i == 0 else ""
                ph = float(r["holm_adjusted_p"])
                p = pfmt(ph)
                if ph < 0.05:
                    p = r"{\boldmath\bfseries " + p + "}"
                ci = f"[{num(float(r['bca_ci_low']), dec, signed=True)}, {num(float(r['bca_ci_high']), dec, signed=True)}]"
                end = r"\\*" if i < len(BASELINES) - 1 else r"\\"
                L.append(f"{lab} & {b} & {ss} & {num(float(r['baseline_mean']), dec)} & "
                         f"{num(float(r['mean_difference']), dec, signed=True)} & {ci} & {p}{end}")
    return "\n".join(L) + "\n"


def extract_body(tex: str) -> str:
    """Rows between the line ``\\endlastfoot`` and ``\\end{longtable}`` of the (single) longtable."""
    i = tex.index(r"\endlastfoot")
    j = tex.index("\n", i) + 1
    k = tex.index(r"\end{longtable}", j)
    return tex[j:k]


# ----------------------------------------------------------------------------------------------- in-text claims
def intext_claims(idx, units_csv: Path | None) -> list[dict]:
    """In-text numbers / statements derived from Table A3, formatted as printed in the paper.

    Each claim is ``{id, paragraph, location, text, ok}``: ``text`` is the string that must occur verbatim in the
    paragraph (``paragraph`` = index into the excerpt file, see :data:`PARAGRAPHS`); ``ok`` is the statistical
    condition the text asserts (e.g. the listed baselines are exactly those with an SSRK decision).  A claim may
    carry an ``info`` entry with related facts that do not enter ``ok``.
    """
    def row(ds, met, b):
        return idx[(ds, met, b)]

    def dec(ds, met):
        return {b: row(ds, met, b)["decision"] for b in BASELINES}

    def f(x, d):
        return f"{float(x):.{d}f}"

    def n_runs(ds=None):
        """The common number of paired runs of all contrasts (of dataset ``ds``); ``None`` if they differ."""
        ns = {int(r["n_pairs"]) for (d, _, _), r in idx.items() if ds is None or d == ds}
        return ns.pop() if len(ns) == 1 else None

    counts = {k: sum(r["decision"] == k for r in idx.values()) for k in ("SSRK", "baseline", "ns")}
    complete = len(idx) == 70 and sum(counts.values()) == len(idx) and n_runs() == 20
    C = []
    # ---- Sec. 5.1 and Abstract: decision counts
    C.append(dict(id="overview_counts", paragraph=0, ok=complete,
                  text=f"Across the {len(idx)} comparisons, SSRK is significantly better in {counts['SSRK']} and "
                       f"worse in {counts['baseline']}; the remaining {counts['ns']} show no significant difference."))
    C.append(dict(id="abstract_counts", paragraph=5, ok=complete,
                  text=f"significantly better than the corresponding baseline in {counts['SSRK']} of {len(idx)} "
                       f"matched {n_runs()}-run comparisons and worse in {counts['baseline']} (Holm adjustment "
                       f"within each dataset--metric family)"))
    # ---- PBMC 3k (Sec. 5.3, "PBMC 3k" paragraph)
    ari = dec("pbmc", "ARI")
    C.append(dict(id="pbmc_ari_mean", paragraph=1, ok=True,
                  text=f"SSRK's ARI ({f(row('pbmc', 'ARI', 'STG')['ssrk_mean'], 3)})"))
    C.append(dict(id="pbmc_ari_sig_higher", paragraph=1,
                  ok=all(ari[b] == "SSRK" for b in ("CAE", "GAEFS", "SSFS", "Variance")),
                  text="is significantly higher than that of CAE, GAEFS, and SSFS (and of Variance"))
    r = row("pbmc", "ARI", "STG")
    C.append(dict(id="pbmc_ari_vs_stg", paragraph=1, ok=ari["STG"] == "ns",
                  text=f"not significantly different from that of STG (${f(r['mean_difference'], 3)}$, 95\\% BCa "
                       f"interval $[{f(r['bca_ci_low'], 3)},{f(r['bca_ci_high'], 3)}]$)"))
    r = row("pbmc", "NMI", "SSFS")
    C.append(dict(id="pbmc_nmi_vs_ssfs", paragraph=1, ok=r["decision"] == "ns",
                  text=f"its NMI is not significantly different from that of SSFS (${f(r['mean_difference'], 3)}$, "
                       f"Holm $p={f(r['holm_adjusted_p'], 2)}$)"))
    C.append(dict(id="pbmc_coverage_all_five", paragraph=1,
                  ok=all(v == "SSRK" for v in dec("pbmc", "CoverageScore").values()),
                  text="its coverage score is significantly higher than that of all five baselines"))
    C.append(dict(id="pbmc_stg_higher_nmi", paragraph=1, ok=row("pbmc", "NMI", "STG")["decision"] == "baseline",
                  text="STG has a higher NMI"))
    r = row("pbmc", "MarkerHits", "SSFS")
    C.append(dict(id="pbmc_ssfs_more_markers", paragraph=1, ok=r["decision"] == "baseline",
                  text=f"SSFS retrieves {f(-float(r['mean_difference']), 1)} more markers on average"))
    # ---- App. A.3.2 "Baselines": PBMC 3k ARI of Variance (mean over the 20 bootstrap refits)
    r = row("pbmc", "ARI", "Variance")
    pbmc_ari = {"SSRK": float(r["ssrk_mean"]), **{b: float(row("pbmc", "ARI", b)["baseline_mean"]) for b in BASELINES}}
    C.append(dict(id="pbmc_variance_ari", paragraph=4,
                  ok=ari["Variance"] == "SSRK" and min(pbmc_ari, key=pbmc_ari.get) == "Variance",
                  text=f"On the per-gene standardized PBMC matrix, Variance is nearly uninformative "
                       f"(ARI {f(r['baseline_mean'], 3)})."))
    # ---- Sec. 5.4, image benchmarks: matched-protocol sentence.  The paragraph is about MNIST (it follows the
    # MNIST five-run comparison of Table 3); the same pattern is reported for Fashion-MNIST as information.
    def image_deviations(ds):
        """Contrasts of ``ds`` whose decision differs from the one the sentence asserts."""
        asserted = {("ARI", b): "ns" if b == "Variance" else "SSRK" for b in BASELINES}
        asserted.update({("NMI", b): "ns" if b == "Variance" else "SSRK" for b in BASELINES})
        asserted.update({("CoverageEff", b): "SSRK" for b in BASELINES})
        return [f"{m} vs {b}: {row(ds, m, b)['decision']} (sentence: {d})"
                for (m, b), d in asserted.items() if row(ds, m, b)["decision"] != d]

    C.append(dict(id="image_matched_mnist", paragraph=3, ok=n_runs("mnist") == 20 and not image_deviations("mnist"),
                  text=f"In the MNIST matched protocol of {n_runs('mnist')} runs, evaluated on pixels only, SSRK's ARI and "
                       f"NMI are significantly higher than those of all four learned baselines and not significantly "
                       f"different from those of Variance, and its coverage efficiency is significantly higher than "
                       f"that of all five baselines.",
                  info={"dataset": "mnist", "fashion_mnist_deviations": image_deviations("fashion_mnist")}))
    # ---- UCI HAR (Sec. 5.5, "UCI HAR" paragraph)
    acc, f1 = dec("uci_har", "Accuracy"), dec("uci_har", "F1_macro")
    C.append(dict(id="uci_acc_f1_means", paragraph=2, ok=True,
                  text=f"its accuracy and macro-F1 ({f(row('uci_har', 'Accuracy', 'STG')['ssrk_mean'], 3)}/"
                       f"{f(row('uci_har', 'F1_macro', 'STG')['ssrk_mean'], 3)})"))
    C.append(dict(id="uci_acc_f1_sig_above", paragraph=2,
                  ok=all(acc[b] == "SSRK" and f1[b] == "SSRK" for b in LEARNED),
                  text="are significantly above those of STG, CAE, GAEFS, and SSFS"))
    r = row("uci_har", "Accuracy", "Variance")
    C.append(dict(id="uci_vs_variance", paragraph=2, ok=acc["Variance"] == "ns" and f1["Variance"] == "ns",
                  text=f"not significantly different from those of Variance (accuracy gap "
                       f"${f(r['mean_difference'], 4)}$, Holm $p={f(r['holm_adjusted_p'], 2)}$)"))
    if units_csv is not None and units_csv.exists():
        with units_csv.open(encoding="utf-8", newline="") as h:
            hni = [float(u["value"]) for u in csv.DictReader(h) if u["dataset"] == "uci_har" and u["metric"] == "HNI"]
        C.append(dict(id="uci_no_appended_column", paragraph=2, ok=len(hni) == 20 and max(hni) == 0.0,
                      text="none of its 20 selections contains an appended column"))
    dbal = dec("uci_har", "DomainBalance")
    C.append(dict(id="uci_domain_lower", paragraph=2,
                  ok=all(dbal[b] == "SSRK" for b in ("Variance", "STG", "CAE", "SSFS")) and dbal["GAEFS"] == "ns",
                  text="its domain imbalance is significantly lower than that of Variance, STG, CAE, and SSFS"))
    for c in C:
        c["location"] = PARAGRAPHS[c["paragraph"]]
    return C


def read_excerpts(path: Path) -> list[str]:
    """Excerpt lines (one verbatim paper source line each); ``%`` comment lines and empty lines are skipped."""
    return [ln.rstrip("\r") for ln in path.read_text(encoding="utf-8").split("\n") if ln and not ln.startswith("%")]


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Body of Table A3 (70 matched contrasts) + comparison with the paper.")
    ap.add_argument("--csv", type=Path, default=PKG / "results" / "paired" / "paired_statistics.csv",
                    help="paired_statistics.csv of paired_stats.py (default results/paired/paired_statistics.csv)")
    ap.add_argument("--units-csv", type=Path, default=None,
                    help="ssrk_per_unit.csv of paired_stats.py (default: next to --csv)")
    ap.add_argument("--out", type=Path, default=None, help="output directory (default: directory of --csv)")
    ap.add_argument("--labels", choices=sorted(LABELS), default="paper", help="row labels: paper (default) or original")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--paper", type=Path, help="full neurips_2026.tex to extract the expected body from (every "
                                              "excerpt of --excerpts must then also occur in it verbatim)")
    g.add_argument("--expected", type=Path, default=PKG / "reference" / "paper" / "paired_longtable_body.tex",
                   help="expected longtable body (default reference/paper/paired_longtable_body.tex)")
    ap.add_argument("--excerpts", type=Path, default=PKG / "reference" / "paired" / "paper_intext_excerpts.tex",
                    help="verbatim paper lines of the in-text claims (default reference/paired/paper_intext_excerpts.tex)")
    a = ap.parse_args(argv)
    out = a.out or a.csv.parent
    out.mkdir(parents=True, exist_ok=True)

    idx = read_rows(a.csv)
    body = longtable_body(idx, a.labels)
    (out / "paired_longtable_body.tex").write_bytes(body.encode("utf-8"))
    paper_tex = None
    if a.paper is not None:
        paper_tex = a.paper.read_bytes().decode("utf-8")
        expected = extract_body(paper_tex)
        src = a.paper
    else:
        expected = a.expected.read_bytes().decode("utf-8")
        src = a.expected
    bytes_identical = body.encode("utf-8") == expected.encode("utf-8")
    ok_table = bytes_identical
    status = "byte-identical" if bytes_identical else "DIFFERENT"
    print(f"longtable body ({len(body.splitlines())} lines) vs {_rel(src)}: {status}")
    if not ok_table:
        el, bl = expected.splitlines(), body.splitlines()
        for n in range(max(len(el), len(bl))):
            e = el[n] if n < len(el) else "<missing>"
            b = bl[n] if n < len(bl) else "<missing>"
            if e != b:
                print(f"  line {n + 1}:\n    paper: {e}\n    ours : {b}")

    units_csv = a.units_csv or (a.csv.parent / "ssrk_per_unit.csv")
    claims = intext_claims(idx, units_csv)
    if not any(c["id"] == "uci_no_appended_column" for c in claims):
        print(f"  [FAIL] uci_no_appended_column: needs the SSRK per-unit values ({_rel(units_csv)} not found)")
    ok_claims = any(c["id"] == "uci_no_appended_column" for c in claims)
    if a.excerpts.exists():
        paras = read_excerpts(a.excerpts)
        if paper_tex is not None:
            src_lines = set(paper_tex.replace("\r\n", "\n").split("\n"))
            verbatim = [p in src_lines for p in paras]
            ok_claims &= all(verbatim)
            print(f"  excerpt lines found verbatim in {_rel(a.paper)}: {sum(verbatim)}/{len(paras)}")
        for c in claims:
            c["in_paper"] = c["paragraph"] < len(paras) and c["text"] in paras[c["paragraph"]]
            ok_claims &= bool(c["in_paper"] and c["ok"])
            print(f"  [{'PASS' if c['in_paper'] and c['ok'] else 'FAIL'}] {c['id']} ({c['location']}): {c['text']}")
            if c["id"] == "image_matched_mnist":
                dev = c["info"]["fashion_mnist_deviations"]
                print("         note: checked for MNIST, the dataset of this paragraph; Fashion-MNIST "
                      + ("shows the same pattern" if not dev else "differs in " + "; ".join(dev)))
    else:
        print(f"excerpt file {_rel(a.excerpts)} not found: in-text claims not compared")
        ok_claims = False
    (out / "intext_claims.json").write_text(json.dumps(claims, indent=1), encoding="utf-8")
    counts = {k: sum(r["decision"] == k for r in idx.values()) for k in ("SSRK", "baseline", "ns")}
    print(f"decisions: SSRK {counts['SSRK']} / baseline {counts['baseline']} / n.s. {counts['ns']}")
    n_claims = len(claims) + (0 if any(c["id"] == "uci_no_appended_column" for c in claims) else 1)
    n_pass = sum(bool(c.get("in_paper") and c["ok"]) for c in claims)
    print(f"table: {'PASS' if ok_table else 'FAIL'}; in-text claims: {'PASS' if ok_claims else 'FAIL'} "
          f"({n_pass}/{n_claims})")
    return 0 if (ok_table and ok_claims) else 1


if __name__ == "__main__":
    sys.exit(main())
