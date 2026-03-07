# spatialxgene

[![CI](https://github.com/murti-abhishek/spatialxgene/actions/workflows/ci.yml/badge.svg)](https://github.com/murti-abhishek/spatialxgene/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/murti-abhishek/spatialxgene/graph/badge.svg)](https://codecov.io/gh/murti-abhishek/spatialxgene)

Interactive spatial transcriptomics viewer for `.h5ad` files — like CellXGene but with spatial coordinates, UMAP, PCA, and scVI embeddings. Includes manual cell selection and differential gene expression (DGE).

## Features

- Visualize spatial, UMAP, PCA, and scVI embeddings
- Color cells by categorical or continuous metadata, or by gene expression
- Datashader-powered rendering for smooth zoom/pan on 500k+ cells
- Point size and opacity controls
- Lasso/box selection for manual cell group definition
- Differential gene expression between two user-defined groups (Welch t-test or Wilcoxon)
- DGE history with per-run summaries
- Dark theme UI

## Installation

```bash
git clone https://github.com/murti-abhishek/spatialxgene
cd spatialxgene
pip install -e .
```

## Usage

```bash
spatialxgene launch my_data.h5ad
```

### Options

```
spatialxgene launch --help

Arguments:
  H5AD_FILE  Path to the .h5ad file.

Options:
  --host TEXT          Host to bind.  [default: 127.0.0.1]
  --port INTEGER       Preferred port (auto-increments if busy).  [default: 8050]
  --subsample N        Randomly subsample to N cells (speeds up large datasets).
  --seed INTEGER       Random seed for subsampling.  [default: 42]
  --debug              Run Dash in debug mode.
  --skip-columns COLS  Comma-separated column names to hide from the Color By dropdown.
```

### Example

```bash
# Full dataset
spatialxgene launch HB_slide_1_scVI_v2.h5ad

# Subsample for faster startup
spatialxgene launch HB_slide_1_scVI_v2.h5ad --subsample 100000

# Hide noisy metadata columns
spatialxgene launch data.h5ad --skip-columns "doublet_score,S_score,G2M_score"
```

## Requirements

- Python >= 3.9
- h5ad files with spatial coordinates in `obsm['spatial']`
- Additional embeddings (`X_umap`, `X_pca`, `X_scVI`) are supported if present

## License

MIT
