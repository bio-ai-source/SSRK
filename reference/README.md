# reference/: what the reruns are compared with

Everything here comes from the original runs that produced the paper or from the final paper: byte copies of the
original records, extracts of them (the baselines' selections, the stability numbers, the merged five-seed evaluator
records), and text taken from the final paper. Exceptions: `exact_gaussian/figureA1_panel_data.csv` is generated
from the records, `uci_har/features.txt` is a copy of a dataset file, and `verification_record/` holds the
from-scratch rerun. `verify.py` and the area scripts read these files. Never edit them: the checks compare files
byte for byte (see `.gitattributes`). Provenance (source path in the original research archive and
SHA-256) is listed in `exact_gaussian/SOURCES.json`, `pbmc/sources.json` and `uci_har/sources.json`.

| Path | Content | Read by | Paper item |
|---|---|---|---|
| `paper/tables/gaussian_{main,tuning,full,ablation}.tex` | the paper's table files | `make_tables.py --check` | Tables 1, A4, A5, A6 |
| `paper/paired_longtable_body.tex` | rows of the 70-contrast longtable | `paired/make_table.py` | Table A3 |
| `paper/figures/FigureA1_Gaussian.pdf` | the paper's figure, for visual comparison | (you) | Figure A1 |
| `exact_gaussian/<run>/` for `eval_main`, `eval_low`, `eval_all`, `ablation_{R1,R5,nosharpen,noentropy,noctx}` | `W_per_seed.npz` (every restart's statistic W and their mean, supports, seeds), `per_seed_q.csv` (selections and FDP/power per dataset and level), `summary.csv`, `per_seed_info.json` (swap checks, norms), `manifest.json` (configuration, seeds, chunk layout, environment, timings) | `run.py compare`, `verify.py`; `run.py rescore` (`verify.py --from-reference`) | Tables 1, A5, A6; Figure A1 |
| `exact_gaussian/tuning/` | per-configuration row files, `tuning_summary.csv`, `selection.json`, `manifest.json` (tuning W were not stored) | same | Table A4 |
| `exact_gaussian/swap_audit/` | 100 paired trajectories x {symmetric (centered), naive} gate x {float32, float64} | same | App. A.4, "Swap checks and compute" |
| `exact_gaussian/figureA1_panel_data.csv` | the plotted values of Figure A1, generated from the records above (not an original record) | `make_figure.py --check` | Figure A1 |
| `pbmc/units/NNN.json`, `NNN_W.npy`, `NNN_Wall.npy` | the 20 bootstrap refits of SSRK: statistic W, selection, ARI, NMI, marker hits, coverage | `pbmc/run.py --check`, `pbmc/rescore.py`, `verify.py` | Table A3 (PBMC rows), Table A8 |
| `pbmc/baseline_selections.csv` | top-100 selections of the five baselines in the 20 refits | `stability.py`, `pbmc/rescore.py --with-baselines` | Table A8 (baseline rows) |
| `pbmc/stability_body_paper.tex`, `pbmc/paper_stability_excerpts.tex`, `pbmc/fill_nr_numbers_stability.json` | rows of Table A8, the paper's stability sentences, the original stability numbers | `stability.py` | Table A8, Sec. 5.3, App. A.5 |
| `uci_har/units/NNN.json`, `NNN_W.npy`, `NNN_Wall.npy` | the 20 training seeds of SSRK: W (mean of 3 restarts), restarts, selections, HNI, domain imbalance | `uci_har/run.py --check`, `uci_har/rescore.py`, `verify.py` | Table A3 (UCI HAR rows) |
| `uci_har/eval5.json` | five-seed MLP accuracy / macro-F1 of each SSRK selection | `evaluate.py --check`, `verify.py` | Table A3 (UCI HAR rows) |
| `uci_har/baseline_selections.csv` | the 384 selected real features of the five baselines per seed | `uci_har/rescore.py --with-baselines` | Table A3 (baseline columns) |
| `uci_har/features.txt` | the 561 feature names of UCI HAR (CC BY 4.0, see `README_features.md`); lets `verify.py --from-reference` run without downloading UCI HAR | `uci_har/rescore.py` | domain imbalance |
| `pbmc/expected_ssrk_means.json`, `uci_har/expected_ssrk_means.json` | the SSRK means of the original paired-statistics record | area scripts | Table A3 |
| `*/orig_resolved_config.json` | the SSRK configurations as resolved by the original code (provenance; asserted by `tests/test_selfrecon.py`) | tests | Table A2 |
| `*/units/orig_config.json` | the original run configuration of the 20 units (unit seeds, overrides); provenance only | (not read) | |
| `paired/paired_v3_final.csv`, `paired/paired_v3_final_means.json` | the original 70-contrast statistics and per-method means | `verify.py` (the CSV), `tests/test_paired_stats.py` (both) | Table A3 |
| `paired/paper_intext_excerpts.tex` | verbatim paper sentences checked by `make_table.py` | `paired/make_table.py` | Abstract, Sec. 5.1-5.5, App. A.3.2 |
| `data/dataset_hashes_release.json` | SHA-256 of the original data files | `tests/test_data_checksums.py` | |
| `verification_record/` | outputs of the from-scratch run on the reference machine (`verify.py --require-all` report, run log) | (you) | |

The per-run records of the baselines and of the image benchmarks, which enter Table A3 as fixed inputs, are in
`experiments/paired/records/` (see `README_records.md` there).
