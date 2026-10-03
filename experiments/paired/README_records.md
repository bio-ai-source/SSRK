# Fixed per-run records used by the paired statistics

`experiments/paired/paired_stats.py` recomputes the 70 matched contrasts of Table A3 of the paper, "Matched
comparison results for all 70 contrasts" (App. A.3.3 "Matched comparison"; LaTeX label `tab:paired_summary`).
Each contrast pairs SSRK with one of the five baselines (Variance, STG, CAE, GAEFS, SSFS) on 20 matched units.
This package re-runs the SSRK side of PBMC 3k and UCI HAR (`experiments/pbmc/run.py`, `experiments/uci_har/run.py`,
`experiments/uci_har/evaluate.py`). Everything else comes from the three fixed records in `records/`:

* **Baselines.** The baselines are not implemented in this package, so their per-run scores are shipped as
  fixed records.
* **MNIST / Fashion-MNIST.** All six methods, SSRK included. The image runs are not retrained here: they were
  produced on a different GPU than the reference machine, and retraining does not reproduce them bitwise.

The values are the original per-run records of the study. They are not re-rounded: every cell is the original
shortest round-trip float string (Python `repr`), so `float(cell)` returns the exact double that the original
analysis used.

| file | rows | content |
|---|---|---|
| `records/pbmc_baselines_per_run.csv` | 100 | PBMC 3k, 5 baselines x 20 bootstrap refits |
| `records/uci_har_baselines_per_run.csv` | 100 | UCI HAR, 5 baselines x 20 training seeds |
| `records/image_per_run.csv` | 240 | MNIST and Fashion-MNIST, 6 methods x 20 training seeds |

## Units and pairing

* PBMC 3k: `bootstrap_id` = 0..19, with bootstrap seed `seed = 20000 + 997 * bootstrap_id`. A row is paired with
  the SSRK unit record `results/pbmc/units/NNN.json` that has `unit == bootstrap_id`. The SSRK record's `seed` must
  equal the bootstrap seed.
* UCI HAR: `seed = 31000 + 997 * unit` for unit 0..19. A row is paired with the SSRK unit record of the same `seed`
  (`results/uci_har/units/NNN.json`, `NNN = unit`).
* MNIST / Fashion-MNIST: `seed = 31000 + 997 * i`, i = 0..19. SSRK and baseline rows of the same `dataset` and
  `seed` are paired.

`paired_stats.py` refuses an SSRK record whose seed is not the evaluation seed of its unit. By default it also
requires all 20 units for every method.

## Columns

Metric definitions are in App. A.8 "Metric definitions" of the paper.

`pbmc_baselines_per_run.csv`: `bootstrap_id, seed, method, ARI, NMI, MarkerHits, CoverageScore`
* `ARI`, `NMI`: the 100 selected genes of the bootstrap resample are reduced by a randomized PCA (at most 50
  components), then clustered with k-means (k = 12, 10 initializations). The partition is compared with the 12
  Leiden clusters (`obs["leiden"]`, resolution 1.5) of the processed file, not with its `celltype*` annotation
  columns.
* `MarkerHits`: number of the 26 curated markers among the 100 selected genes (an integer, written as in the
  original file, e.g. `14`).
* `CoverageScore`: marker hits divided by the number of lineage groups with at least one selected marker.

`uci_har_baselines_per_run.csv`: `seed, unit, method, Accuracy, F1_macro, HNI, DomainBalance`
* `Accuracy`, `F1_macro`: test accuracy and macro-F1 of the deterministic MLP evaluator, averaged over its five
  fixed initializations ("UCI HAR accuracy and macro-F1 are averaged over five MLP initializations"). The
  evaluator is applied to each baseline's stored selection of the 384 top-ranked real features: the top 384
  among the 561 original features, with appended columns ineligible. The evaluator seeds are
  `seed + 41 + 1000 k`, k = 0..4.
* `HNI`: fraction of the top 384 of all 1,122 ranked columns (561 features + 561 appended within-column
  permutations) that are appended columns.
* `DomainBalance`: the group-normalized domain imbalance `|n_time/272 - n_freq/289|` of the 384 real-feature
  selection, printed as "domain imbalance" in the paper.

`image_per_run.csv`: `dataset, seed, method, ARI, NMI, CoverageEff`
* `dataset`: `mnist` (top 256 pixels) or `fashion_mnist` (top 128 pixels).
* `ARI`, `NMI`: the selected pixels are reduced by a randomized PCA (at most 50 components), then clustered with
  k-means (k = 10, 10 initializations); the partition is compared with the digit/garment class labels. This is
  the PBMC 3k procedure with k = 10.
* `CoverageEff`: coverage efficiency, including the purity weight. In this protocol every selected column is a
  pixel, so the purity weight is 1.

## Provenance (not needed for reproduction)

`records/build_records.py` is a provenance tool. It documents how the three files were extracted from the
original research archive, and it runs only with that archive (`--orig <archive root>`), which is not part of this
package. Reproduction never calls it: `paired_stats.py` reads the shipped CSV files. Given the archive, the script
re-derives the three files byte for byte. The source paths below are relative to the archive root.

| shipped file (sha256) | source (sha256) | transformation |
|---|---|---|
| `pbmc_baselines_per_run.csv` (`da5c5c1b393972d30eecfc6e9291be254722e1e3d4a5460ae340a8ace92b7444`) | `revised/results/local_minimal_v2_exact/01_pbmc_exact_top100_per_run.csv` (`12d61fc9ebc7706f480887fef999b8ed2119d698e0ebe25265a26f80d87c8761`) | keep rows with `status == ok` and `method != SSRK`; keep the listed columns verbatim |
| `uci_har_baselines_per_run.csv` (`1c59a79dc755a4cb7b4b415833a0d51166337c67ef126a926de965669e82a747`) | `revised/results/local_minimal_v2_exact/03_uci_exact_per_run.csv` (`58fd6d5cc33adcbf1c4a2b0144e986f8743f0fe7e48712a117765b09141da003`) and `revised/cr2/results/uci_eval5/stored_00.json` ... `stored_07.json` | keep rows with `status == ok` and `method != SSRK`; HNI and DomainBalance verbatim; Accuracy and F1_macro replaced by the five-seed evaluator values of the stored selection (`stored_*.json`, key `f"{unit:03d}_{method}"`), written with `repr` |
| `image_per_run.csv` (`5f0a1a7a20aa198e6e2b5cd4cce6e01dd2afa7f8bb6215ed39535174cbe6bf8e`) | `revised/results/local_minimal_v2_exact/02_image_exact_per_run.csv` (`909efe18ce6e8433ffd2a714148579b1f93d5b360017c57d19e9ce899b3cb9df`) | keep rows with `status == ok` (all six methods); keep the listed columns verbatim |

SHA-256 values of the `stored_*.json` sources, in order 00 to 07:
`e99c3dae249e0042e61152a92a8ae30a22bfb491c0ff16bcae564350b73ff4c3`,
`a6a6697c2a15116c59e333deba370f3862a14dbcc05c569458847fa4012c01ac`,
`6bc4fb191de27c7457935f3b1f198bd0480a447f858169fa85c4362f081f4bcb`,
`5821d821a29041a1152baa75aff603ff2265e645f5e28762ce6c6d3e94cfe947`,
`f968f9175a63eb1ecc9454a32e4c4812e0910b3b62c5ce4f6449f144089188f4`,
`648bb03ed9f43a3705ae45838fb3dd4b1ac14e118dec5149712a79c90b0b90a6`,
`dc4b313bf7139c99825b666688c9df7639e5a8071d44a793edf52ad31fd28343`,
`677915fc79188d4a26018c9e495a1d22a5121b055d3349fc6741d7701a0c0291`.

The bulky columns of the original files (`selected_indices`, `selection_size`, `runtime_seconds`, `status`,
`error`, and for images `Stability` and `HNI_diagnostic`) are dropped from these three files. None of them enters
the paired table. The baselines' selections of PBMC 3k and UCI HAR are shipped separately, in
`reference/pbmc/baseline_selections.csv` and `reference/uci_har/baseline_selections.csv`. From them,
`experiments/pbmc/rescore.py --with-baselines` and `experiments/uci_har/rescore.py --with-baselines` recompute the
baseline metrics of the first two files (UCI HAR accuracy and macro-F1 also need `--with-evaluator` and the UCI HAR
data). Baseline HNI needs the full ranking of all 1,122 columns, which is not shipped, so it stays a record.

## Expected outputs

Applied to SSRK records that equal the original ones, `paired_stats.py` gives:

* `results/paired/paired_statistics.csv`, byte-identical to the original `paired_v3_final.csv` (copy in
  `reference/paired/`): 70 rows; columns `dataset, metric, contrast, n_pairs, ssrk_mean, baseline_mean,
  mean_difference, bca_ci_low, bca_ci_high, paired_randomization_p, holm_adjusted_p, decision`.
* `results/paired/paired_means.json`, byte-identical to `paired_v3_final_means.json`.
* 52 SSRK / 4 baseline / 14 n.s. decisions.

`make_table.py` then typesets the longtable body. That body is byte-identical to
`reference/paper/paired_longtable_body.tex`, the paper's rows between `\endlastfoot` and `\end{longtable}`.
`make_table.py` also checks the 16 statements of the paper that are read off this table (Abstract; Sec. 5.1,
5.3, 5.4 and 5.5; App. A.3.2 "Baselines") against the verbatim source lines in
`reference/paired/paper_intext_excerpts.tex`, and writes them with their outcome to `intext_claims.json`.
