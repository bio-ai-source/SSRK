# Notice: PBMC 3k data file

This directory redistributes one third-party data file, unmodified.

| | |
|---|---|
| file | `pbmc3k_processed.h5ad` (52,633,500 bytes) |
| SHA-256 | `52f5becad785656ab391d2b41a91bf71344a1f0a39e19d510e61668d6038ad0f` |
| dataset | "PBMC 3k test datasets for besca", version 1.0 (published 2020-04-15) |
| creators | Klas Hatje, Alice Julien-Laferrière |
| source | Zenodo record 3886414, <https://zenodo.org/records/3886414> |
| DOI | [10.5281/zenodo.3886414](https://doi.org/10.5281/zenodo.3886414) |
| license | **AGPL-3.0-only**, as declared on the Zenodo record; the full license text is in `LICENSE` in this directory |
| modifications | none: the file is redistributed exactly as published (it has the MD5 `1a39f94c6a62a088941a5e4296f31f39` listed on the Zenodo record) |
| underlying data | PBMC 3k dataset of 10x Genomics (Zheng et al., 2017, *Nature Communications* 8, 14049, doi:[10.1038/ncomms14049](https://doi.org/10.1038/ncomms14049)), reprocessed by the creators with the besca package; see the 10x Genomics dataset page (<https://www.10xgenomics.com/datasets>) for the terms of the underlying data |

The AGPL-3.0-only license applies to this data file only. It is not the license of the SSRK code in this
package.

`python data/download.py --pbmc --check` verifies the file against `data/checksums.json`; `data/README.md`
describes its content and how the experiments read it.

When using the file, please cite:

* K. Hatje and A. Julien-Laferrière. PBMC 3k test datasets for besca, version 1.0. Zenodo, 2020.
  doi:10.5281/zenodo.3886414.
* G. X. Y. Zheng et al. Massively parallel digital transcriptional profiling of single cells.
  *Nature Communications* 8, 14049, 2017. doi:10.1038/ncomms14049.
