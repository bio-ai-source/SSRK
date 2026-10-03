# Data

The in-scope real-data experiments of the package use two public datasets:

| dataset | used by | shipped? | size | how to obtain |
|---|---|---|---|---|
| PBMC 3k, processed AnnData (`pbmc3k_processed.h5ad`) | `experiments/pbmc/` (matched protocol, Table A8 stability) | **yes**: `data/pbmc/` | 52.6 MB | nothing to do; `python data/download.py --pbmc` checks the file and downloads it again if it is missing or corrupted |
| UCI HAR (Human Activity Recognition Using Smartphones) | `experiments/uci_har/` (matched protocol, five-seed evaluator) | **no** (282.6 MB = 269.5 MiB extracted) | 61.0 MB download | `python data/download.py --uci-har` |

The exact Gaussian study (`experiments/exact_gaussian/`) generates its own data and needs no files.
MNIST and Fashion-MNIST are not needed. The package does not retrain the image benchmarks, and the
paired statistics use their shipped per-run records.

## Quick start

From the package root:

```bash
python data/download.py --all      # check PBMC (repair if needed) + download, extract and verify UCI HAR
python data/download.py --check    # verify everything, offline (exit status 0 = all files verified)
python -m pytest tests/test_data_checksums.py -q
```

`download.py` uses only the Python standard library (`urllib`) and follows `HTTPS_PROXY`/`HTTP_PROXY`.
Because the PBMC file is shipped and verifies, `--all` downloads only the 61.0 MB UCI HAR archive;
depending on server speed this takes about 1 to 7 min. The 52.6 MB PBMC file is downloaded again
from Zenodo only if it is missing or corrupted, which adds about 1.5 to 11 min.
Extraction plus verification takes about 2 s. A repeated run finds every file verified and finishes
in under 1 s without network access.

## Verification guarantees

`data/checksums.json` lists the size and SHA-256 of every file the experiments read. The digests
are those of the files that produced the paper's numbers, as recorded in the original study's
`dataset_hashes_release.json` (copy: `reference/data/dataset_hashes_release.json`;
`tests/test_data_checksums.py` checks that the two agree). The downloader enforces them as follows:

* Data are streamed into `<file>.part`. A file is moved to its final name (`os.replace`) only after
  its digest matches, so an interrupted or corrupted transfer never leaves a truncated file under a
  final name.
* Interrupted downloads are resumed with HTTP `Range` requests when the server supports them
  (Zenodo does). Downloads are retried with exponential back-off, honour `Retry-After` on HTTP
  429/503 and fall back to mirror URLs.
* The script is idempotent: files that are already present and verified are skipped.
* For the PBMC file, a digest mismatch is an error. For UCI HAR, the recorded archive digest is
  informational only. If the UCI repository re-packages the archive, a warning
  is printed and the extracted files are still checked one by one; only those digests decide success.
* Only the files listed in the manifest are extracted, to paths built from the manifest, never from
  member names inside the archive. Junk entries such as `__MACOSX/` and `.DS_Store` are ignored.

`download.py` options:

| option | effect |
|---|---|
| `--all` (default), `--pbmc`, `--uci-har` | choose the datasets |
| `--check` | verify only; never accesses the network |
| `--data-dir DIR` | data root (default: this directory) |
| `--archive-dir DIR` | where the downloaded archives are kept (default `DIR/_downloads`, safe to delete afterwards) |
| `--uci-full` | also extract and verify the 18 raw `Inertial Signals` files (not used; about 190 MB) |
| `--force` | re-extract UCI HAR files even if they already verify |
| `--retries N`, `--timeout S`, `--quiet` | network and logging settings |

## Layout expected by the experiments

```
data/
  checksums.json, download.py, prepare_uci_har.py, README.md
  pbmc/pbmc3k_processed.h5ad                 shipped
  pbmc/LICENSE, pbmc/NOTICE.md               license text (AGPL-3.0) and notice of the PBMC file (shipped)
  uci_har/UCI HAR Dataset/                    created by download.py / prepare_uci_har.py (not shipped)
      features.txt  activity_labels.txt  README.txt  features_info.txt
      train/X_train.txt  train/y_train.txt  train/subject_train.txt
      test/X_test.txt    test/y_test.txt    test/subject_test.txt
  uci_har/uci_har_arrays.npz                  optional cache (prepare_uci_har.py --cache; not shipped)
  _downloads/                                 archive cache of download.py (not shipped; may be deleted)
```

Every script that reads these files takes `--data-dir` (default: this directory). The loaders raise an error if a
file is missing. They never substitute placeholder data.

### SHA-256 of the files read by the experiments

| path (relative to `data/`) | bytes | SHA-256 |
|---|---:|---|
| `pbmc/pbmc3k_processed.h5ad` | 52,633,500 | `52f5becad785656ab391d2b41a91bf71344a1f0a39e19d510e61668d6038ad0f` |
| `uci_har/UCI HAR Dataset/features.txt` | 15,785 | `b384889885b1c8680ecf250e0e605b4c79536cdf09ce9f2fa7225a3c6773b5c7` |
| `uci_har/UCI HAR Dataset/activity_labels.txt` | 80 | `f6e6b292704438261f0283b2a82659ab0661447dc16520938bdf379ed1bf8de0` |
| `uci_har/UCI HAR Dataset/train/X_train.txt` | 66,006,256 | `9c1246099c8d5463779eec5a3845cc3898df9d8c715441ab2a1232d4421fc42f` |
| `uci_har/UCI HAR Dataset/train/y_train.txt` | 14,704 | `415b3357e2e2d5e70a45c781947c6bdd80e0410a579d862eedcd6bbce3c694ef` |
| `uci_har/UCI HAR Dataset/train/subject_train.txt` | 20,152 | `66e1e2fbe9b396fa5155f531dc3348c803f503be663e2318772d49947db9153a` |
| `uci_har/UCI HAR Dataset/test/X_test.txt` | 26,458,166 | `99035209add50d17650e847d8a4658d784a5707466ebb7c2deb97c28e5250a78` |
| `uci_har/UCI HAR Dataset/test/y_test.txt` | 5,894 | `2a94cb53ca46b956c1b2aebd4a09a72badd8a8e9ea9cdfe9f31b3306a08666bb` |
| `uci_har/UCI HAR Dataset/test/subject_test.txt` | 7,934 | `63b8c577f4a85431c06bd87f6384608c90a9f72f05c7956dd21255dc63515c07` |
| `uci_har/UCI HAR Dataset/README.txt` | 6,304 | `eaee53f45825f349681a1a1b92a997a0082648f9b5bf93ddb3290caeede4b3a8` |
| `uci_har/UCI HAR Dataset/features_info.txt` | 2,809 | `5d7e7d1f204dcfba846043029ef7a15a07c2b3711ee0a122da031f5971b3b585` |

`checksums.json` also lists the digests of the 18 `Inertial Signals` files, which are not used, and
of the two UCI archives.

---

## PBMC 3k (processed, besca test dataset)

**What it is.** About 3,000 peripheral blood mononuclear cells from a healthy donor, profiled with
10x Genomics Chromium single-cell RNA-seq (the "PBMC 3k" dataset of 10x Genomics; Zheng et al. 2017).
The data were taken from the Seurat 3k PBMC tutorial and reprocessed with the besca package
(Roche). The result was published as "PBMC 3k test datasets for besca". This package uses the
*processed* file of that record: 2,504 annotated cells and 1,719 genes, per-gene z-scored and
clipped at 10. The cell and gene filtering, normalisation and clustering were all done upstream by
besca; this package does not recompute any of it.

| | |
|---|---|
| source | Zenodo record 3886414, version 1.0 (2020-04-15), doi:[10.5281/zenodo.3886414](https://doi.org/10.5281/zenodo.3886414) |
| file URL | <https://zenodo.org/records/3886414/files/pbmc3k_processed.h5ad?download=1> |
| size | 52,633,500 bytes; MD5 `1a39f94c6a62a088941a5e4296f31f39` (as published on Zenodo; it matches the shipped file) |
| SHA-256 | `52f5becad785656ab391d2b41a91bf71344a1f0a39e19d510e61668d6038ad0f` |
| license | The Zenodo record declares **AGPL-3.0-only**. The file is redistributed here unmodified with attribution, and that license applies to this file only, independently of the license of the code. The full license text is in `data/pbmc/LICENSE` (canonical text from <https://www.gnu.org/licenses/agpl-3.0.txt>), and `data/pbmc/NOTICE.md` records title, creators, DOI and SHA-256 of the file. The underlying PBMC 3k data are published by 10x Genomics; see the 10x Genomics dataset page (<https://www.10xgenomics.com/datasets>) for their terms. |

**What the experiments read.** The file is read with `anndata` (`experiments/pbmc/data.py`). The
result is bitwise identical to the original study's scanpy-based loader
(`load_pbmc_data(source="processed", n_hvgs=1719, zscore_clip=None)`):

* `X = np.asarray(adata.X, dtype=np.float32)`: dense, shape (2504, 1719), used as stored. The
  per-gene means lie between -0.025 and 2e-9. The per-gene standard deviations lie between 0.28 and
  1.0; the clipping pulls them below 1. The maximum value is exactly 10.0, reached by 2,916 entries in
  1,038 genes, and the minimum is -2.865. The code applies no further normalisation, gene filtering
  or clipping.
* `labels = adata.obs["leiden"].cat.codes` (int8): the original loader uses `obs["louvain"]` if it
  exists and `obs["leiden"]` otherwise. This file has no `louvain` column, so the labels are the
  **12 Leiden clusters** `'0'`…`'11'`. They were computed by besca at resolution 1.5 with
  random_state 0 (`uns["leiden"]["params"]`). The categories are stored in the order `'0'`…`'11'`,
  so code *k* is cluster *k*. Cluster sizes are 438, 317, 257, 245, 234, 212, 191, 159, 153, 153,
  116 and 29. The labels are used only for evaluation (ARI/NMI against *k*-means with *k* = 12),
  never for training.
* `gene_names = adata.var_names` (1,719 unique gene symbols, e.g. `ISG15`, `TNFRSF4`, …). These are
  used to locate the curated marker genes (MarkerHits / CoverageScore).
* The other content of the file is not used: `.raw` (2504 × 14702), the cell-type annotations and
  the embeddings.

Each of the 20 matched-protocol units draws a row bootstrap of `X` with
`np.random.default_rng(seed).choice(2504, 2504, replace=True)` (paper App. A.3.3 "Matched comparison").

**Manual download.** Download the file from the URL above, place it at
`data/pbmc/pbmc3k_processed.h5ad`, and run `python data/download.py --pbmc --check`.

---

## UCI HAR (Human Activity Recognition Using Smartphones)

**What it is.** 30 volunteers performed six activities (walking, walking upstairs, walking
downstairs, sitting, standing, lying) with a waist-mounted Samsung Galaxy S II. The dataset provides
561 engineered time- and frequency-domain features per 2.56 s window, computed by the dataset
creators from the accelerometer and gyroscope signals and scaled to [-1, 1]. The official split
assigns 21 subjects (7,352 windows) to training and 9 subjects (2,947 windows) to testing.

| | |
|---|---|
| source | UCI Machine Learning Repository, dataset id 240: <https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones>, doi:[10.24432/C54S4K](https://doi.org/10.24432/C54S4K) |
| download URL | <https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip> |
| archive | 61,005,872 bytes, SHA-256 `c00b803081a5c797cd5e4b83700a9810b38d53d9d84e01917e090e1fdbc81031` (official download; informational). It contains `UCI HAR Dataset.names` and the inner archive `UCI HAR Dataset.zip` (60,999,314 bytes, SHA-256 `2045e435c955214b38145fb5fa00776c72814f01b203fec405152dac7d5bfeb0`). |
| extracted size | 282.6 MB for all 28 dataset files; 92.5 MB for the 10 files extracted by default |
| license | **CC BY 4.0**, as stated on the UCI repository page. The creators' own `README.txt` asks that publications cite Anguita et al. (ESANN 2013). It also states that the dataset is distributed as-is and that commercial use is prohibited. |

**What the experiments read** (`experiments/uci_har/data.py`; the original `load_uci_har`):

```python
X_train = np.loadtxt("UCI HAR Dataset/train/X_train.txt", dtype=np.float32)   # (7352, 561)
y_train = np.loadtxt("UCI HAR Dataset/train/y_train.txt", dtype=int) - 1       # 0..5
X_test  = np.loadtxt("UCI HAR Dataset/test/X_test.txt",  dtype=np.float32)     # (2947, 561)
y_test  = np.loadtxt("UCI HAR Dataset/test/y_test.txt",  dtype=int) - 1
names   = [line.split(maxsplit=1)[1] for line in open("UCI HAR Dataset/features.txt")]
```

* The official split and the 561 supplied features are used without further standardisation. The
  only change is that labels are shifted to start at 0.
* `features.txt` defines the domain-imbalance metric: names starting with `f` are frequency-domain
  features (289). All other names count as time-domain (272: the 265 `t…` features plus the 7
  `angle(…)` features).
* `train/subject_train.txt` and `test/subject_test.txt` are verified but enter no computation of this package
  (`subject_train.txt` was used only by the subject-wise validation split of the original tuning runs, which are not
  part of this package).
  `test/subject_test.txt` and `activity_labels.txt` are official split metadata and class names.
  They are verified but do not enter any computation.
* For the HNI diagnostic, the experiments append 561 within-column-permuted copies of training
  columns in memory (`default_rng(seed + 7)`). This is done in code; no derived file is written.

**Manual download (`prepare_uci_har.py`).** If `download.py` cannot reach the repository, for example
because of a firewall:

1. Download `human+activity+recognition+using+smartphones.zip` in a browser from the URL above. The
   "Download" button of the dataset page links to the same URL.
2. Run `python data/prepare_uci_har.py --zip path/to/human+activity+recognition+using+smartphones.zip`.
   The inner `UCI HAR Dataset.zip` is accepted as well. The script extracts the files listed above
   into `data/uci_har/UCI HAR Dataset/` and verifies them; this took about 1 to 4 s on the
   reference machine. With `--uci-full` it also extracts the raw inertial signals.
3. Optional: `python data/prepare_uci_har.py --cache` parses the text files once and writes
   `data/uci_har/uci_har_arrays.npz` (19.2 MB, compressed). The file holds `X_train`, `X_test`
   (float32), the 0-based `y_train` and `y_test`, `subject_train`, `subject_test`, `feature_names`,
   `activity_labels`, and the SHA-256 of the source files. After writing, every array is re-read and
   compared byte for byte with `np.loadtxt` of the text files. Use `--verify-cache` to repeat that
   check later. **The experiments always read the text files**; the cache is only a convenience for
   interactive work.

If the files are already extracted elsewhere, copy the `UCI HAR Dataset` folder to `data/uci_har/`
and run `python data/prepare_uci_har.py` to verify it.

---

## Citations

```bibtex
@misc{hatjeBescaPBMC3k2020,
  author = {Hatje, Klas and Julien-Laferri{\`e}re, Alice},
  title  = {{PBMC} 3k test datasets for besca},
  year   = {2020}, publisher = {Zenodo}, version = {1.0},
  doi    = {10.5281/zenodo.3886414}
}
@article{zhengMassivelyParallelDigital2017,
  title   = {Massively parallel digital transcriptional profiling of single cells},
  author  = {Zheng, Grace X. Y. and Terry, Jessica M. and Belgrader, Phillip and others},
  journal = {Nature Communications}, volume = {8}, pages = {14049}, year = {2017},
  doi     = {10.1038/ncomms14049}
}
@misc{reyesortizUCIHAR2013,
  author = {Reyes-Ortiz, Jorge and Anguita, Davide and Ghio, Alessandro and Oneto, Luca and Parra, Xavier},
  title  = {Human Activity Recognition Using Smartphones},
  year   = {2013}, howpublished = {UCI Machine Learning Repository},
  doi    = {10.24432/C54S4K}
}
@inproceedings{anguitaPublicDomainDataset2013,
  title     = {A Public Domain Dataset for Human Activity Recognition Using Smartphones},
  author    = {Anguita, Davide and Ghio, Alessandro and Oneto, Luca and Parra, Xavier and Reyes-Ortiz, Jorge Luis},
  booktitle = {European Symposium on Artificial Neural Networks, Computational Intelligence and Machine Learning (ESANN)},
  year      = {2013}, pages = {437--442}
}
```
