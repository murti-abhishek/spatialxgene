"""Data loading for spatialxgene — reads h5ad directly via h5py."""

from __future__ import annotations

import math
import re
import h5py
import numpy as np
import pandas as pd
import plotly.express as px
from pathlib import Path
from typing import Optional

_GENE_CACHE_MAX = 50  # max genes cached in memory (~230 MB at 580 k cells)

_SKIP_COLS = {
    '_index', 'region', 'segmentation_method', 'z_level',
    'control_codeword_counts', 'control_probe_counts',
    'deprecated_codeword_counts', 'genomic_control_counts',
    'unassigned_codeword_counts', '_scvi_batch', '_scvi_labels',
    'pct_counts_in_top_10_genes', 'pct_counts_in_top_20_genes',
    'pct_counts_in_top_50_genes', 'pct_counts_in_top_150_genes',
}

# Candidate obs column names for a library/sample identifier
_LIB_ID_CANDIDATES = ['library_id', 'sample_id', 'sample', 'batch', 'slide_id', 'library']

_VIEW_KEYS = [
    ('Spatial', 'spatial'),
    ('UMAP',    'X_umap'),
    ('PCA',     'X_pca'),
    ('scVI',    'X_scVI'),
]

_AXIS_LABELS = {
    'spatial': ('X (µm)', 'Y (µm)'),
    'umap':    ('UMAP 1', 'UMAP 2'),
    'pca':     ('PC 1',   'PC 2'),
    'scvi':    ('scVI 1', 'scVI 2'),
}

# Candidate obs column pairs for synthesising obsm['spatial'] when absent
_SPATIAL_COL_PAIRS = [
    ('center_x',  'center_y'),
    ('x_centroid','y_centroid'),
    ('spatial_x', 'spatial_y'),
    ('x',         'y'),
]


def _decode_bytes(arr: np.ndarray) -> list:
    return [x.decode() if isinstance(x, bytes) else x for x in arr]


def _bh_correction(pvals: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR correction (vectorised)."""
    n = len(pvals)
    if n == 0:
        return pvals.copy()
    order = np.argsort(pvals)
    ranks = np.arange(1, n + 1, dtype=float)
    padj_sorted = np.minimum(pvals[order] * n / ranks, 1.0)
    # Enforce monotonicity: cumulative minimum from right
    padj_sorted = np.minimum.accumulate(padj_sorted[::-1])[::-1]
    padj = np.empty(n)
    padj[order] = padj_sorted
    return padj


class SpatialData:
    """In-memory store for a single h5ad file."""

    def __init__(self, path: str, subsample: Optional[int] = None, seed: int = 42,
                 skip_columns: Optional[set] = None, shift_libraries: bool = True):
        self.path = Path(path)
        self.name = self.path.stem
        self._subsample_n = subsample
        self._seed = seed
        self._skip_cols = _SKIP_COLS if skip_columns is None else set(skip_columns)
        self._shift_libraries = shift_libraries
        self._load()

    def _load(self):
        with h5py.File(self.path, 'r') as f:
            obs_grp = f['obs']
            # The '_index' *attribute* stores the name of the index dataset.
            # Older files store the index directly as a dataset named '_index';
            # newer AnnData (encoding-version 0.2+) uses an attribute to name it.
            _idx_key = obs_grp.attrs.get('_index', '_index')
            if _idx_key not in obs_grp:
                _idx_key = '_index'  # fall back to literal key
            n_total = obs_grp[_idx_key].shape[0]
            self.n_cells_total = n_total

            rng = np.random.default_rng(self._seed)
            if self._subsample_n and self._subsample_n < n_total:
                self._idx = np.sort(rng.choice(n_total, self._subsample_n, replace=False))
            else:
                self._idx = None  # keep all

            idx = self._idx

            # --- obsm ---
            self.obsm: dict[str, np.ndarray] = {}
            for key in f['obsm'].keys():
                try:
                    arr = f['obsm'][key][:]
                    self.obsm[key] = arr if idx is None else arr[idx]
                except Exception:
                    pass

            # --- obs ---
            obs_raw: dict = {}
            raw_ids = obs_grp[_idx_key][:]
            cell_ids = _decode_bytes(raw_ids) if raw_ids.dtype.kind in ('S', 'O') else list(raw_ids)
            if idx is not None:
                cell_ids = [cell_ids[i] for i in idx]

            for col in f['obs'].keys():
                if col == '_index':
                    continue
                try:
                    item = f['obs'][col]
                    if isinstance(item, h5py.Dataset):
                        data = item[:]
                        if idx is not None:
                            data = data[idx]
                        if data.dtype.kind in ('S', 'O'):
                            data = _decode_bytes(data)
                        obs_raw[col] = data
                    elif isinstance(item, h5py.Group):
                        if 'codes' in item and 'categories' in item:
                            codes = item['codes'][:]
                            cats  = item['categories'][:]
                            if idx is not None:
                                codes = codes[idx]
                            cats = _decode_bytes(cats) if cats.dtype.kind in ('S', 'O') else list(cats)
                            obs_raw[col] = pd.Categorical.from_codes(codes, categories=cats)
                except Exception:
                    pass

            self.obs = pd.DataFrame(obs_raw, index=cell_ids)
            self.n_cells = len(self.obs)

            # Synthesise obsm['spatial'] from obs columns when absent
            if 'spatial' not in self.obsm:
                for xcol, ycol in _SPATIAL_COL_PAIRS:
                    if xcol in obs_raw and ycol in obs_raw:
                        xarr = np.asarray(obs_raw[xcol], dtype=float)
                        yarr = np.asarray(obs_raw[ycol], dtype=float)
                        self.obsm['spatial'] = np.column_stack([xarr, yarr])
                        break

            # --- uns colors ---
            self.uns_colors: dict[str, list[str]] = {}
            if 'uns' in f:
                for key in f['uns'].keys():
                    if not key.endswith('_colors'):
                        continue
                    col_name = key[:-7]
                    try:
                        item = f['uns'][key]
                        if isinstance(item, h5py.Dataset):
                            raw = item[:]
                            self.uns_colors[col_name] = [
                                c.decode() if isinstance(c, bytes) else c for c in raw
                            ]
                    except Exception:
                        pass

            # --- gene names ---
            self.gene_names: list[str] = []
            self._gene_idx_map: dict[str, int] = {}
            if 'var' in f:
                try:
                    raw = f['var']['_index'][:]
                    self.gene_names = _decode_bytes(raw) if raw.dtype.kind in ('S', 'O') else [str(x) for x in raw]
                    self._gene_idx_map = {g: i for i, g in enumerate(self.gene_names)}
                except Exception:
                    pass

        # Gene expression cache (populated lazily)
        self._gene_cache: dict[str, np.ndarray] = {}
        self._csc_matrix = None   # column-sparse  — efficient per-gene access
        self._csr_matrix = None   # row-sparse     — efficient per-cell access for DGE

        # Shift overlapping library spatial coordinates into a grid layout
        if self._shift_libraries:
            self._shift_library_spatial_coords()

    # ------------------------------------------------------------------
    # Multi-library spatial layout
    # ------------------------------------------------------------------

    def _shift_library_spatial_coords(self) -> None:
        """Grid-arrange per-library spatial sections when their bboxes overlap.

        Detects a library-ID column in obs (e.g. ``library_id``, ``sample``),
        checks whether the sections' bounding boxes intersect, and — if they do
        — shifts each library's XY coordinates so they are laid out side-by-side
        in a grid with 10 % padding.  All non-spatial embeddings (UMAP, PCA …)
        are left untouched.
        """
        if 'spatial' not in self.obsm:
            return

        # Find the library-ID column
        lib_col: Optional[str] = None
        for cand in _LIB_ID_CANDIDATES:
            if cand in self.obs.columns:
                lib_col = cand
                break
        if lib_col is None:
            return

        lib_arr = np.asarray(self.obs[lib_col], dtype=str)   # always plain str
        libraries = sorted(set(lib_arr))
        if len(libraries) < 2:
            return

        coords = self.obsm['spatial']   # view into the array — we'll replace it

        # Per-library bounding boxes
        bboxes: dict[str, dict] = {}
        for lib in libraries:
            mask = lib_arr == lib
            pts = coords[mask]
            if len(pts) == 0:
                continue
            bboxes[lib] = {
                'mask': mask,
                'xmin': float(pts[:, 0].min()),
                'xmax': float(pts[:, 0].max()),
                'ymin': float(pts[:, 1].min()),
                'ymax': float(pts[:, 1].max()),
                'w':    float(pts[:, 0].max() - pts[:, 0].min()),
                'h':    float(pts[:, 1].max() - pts[:, 1].min()),
            }

        lib_list = [l for l in libraries if l in bboxes]
        if len(lib_list) < 2:
            return

        # Check whether any pair of bounding boxes overlaps
        def _overlaps(a: dict, b: dict) -> bool:
            return not (
                a['xmax'] < b['xmin'] or b['xmax'] < a['xmin'] or
                a['ymax'] < b['ymin'] or b['ymax'] < a['ymin']
            )

        has_overlap = any(
            _overlaps(bboxes[lib_list[i]], bboxes[lib_list[j]])
            for i in range(len(lib_list))
            for j in range(i + 1, len(lib_list))
        )
        if not has_overlap:
            return  # sections are already spatially disjoint — leave as-is

        # Arrange in a grid: ceil(sqrt(N)) columns, enough rows to fit all
        n = len(lib_list)
        n_cols = math.ceil(math.sqrt(n))

        max_w = max(bboxes[l]['w'] for l in lib_list)
        max_h = max(bboxes[l]['h'] for l in lib_list)
        pad_x = max_w * 0.10
        pad_y = max_h * 0.10

        new_coords = coords.copy().astype(float)
        for idx, lib in enumerate(lib_list):
            b = bboxes[lib]
            col_pos = idx % n_cols
            row_pos = idx // n_cols
            # Translate so this library's own origin is (0,0), then apply grid offset
            x_shift = col_pos * (max_w + pad_x) - b['xmin']
            y_shift = row_pos * (max_h + pad_y) - b['ymin']
            new_coords[b['mask'], 0] += x_shift
            new_coords[b['mask'], 1] += y_shift

        self.obsm['spatial'] = new_coords
        print(
            f"[spatialxgene] {n} libraries detected in '{lib_col}'; "
            f"spatial sections shifted into {math.ceil(n / n_cols)}×{n_cols} grid."
        )

    # ------------------------------------------------------------------
    # Views / embeddings
    # ------------------------------------------------------------------

    def available_views(self) -> list[dict]:
        seen: set = set()
        result = []
        # Known views first (preserves display order)
        for label, obsm_key in _VIEW_KEYS:
            if obsm_key in self.obsm:
                result.append({'label': label, 'value': label.lower()})
                seen.add(obsm_key)
        # Auto-detect any remaining 2-D obsm arrays
        for key, arr in self.obsm.items():
            if key not in seen and arr.ndim == 2 and arr.shape[1] >= 2:
                result.append({'label': key, 'value': key})
        return result

    def get_coords(self, view: str, flip_y: bool = False, flip_x: bool = False):
        key_map = {label.lower(): obsm_key for label, obsm_key in _VIEW_KEYS}
        # Known views map to their obsm key; unknown views use the raw value as the key
        obsm_key = key_map.get(view.lower(), view)
        if obsm_key in self.obsm:
            arr = self.obsm[obsm_key]
            x, y = arr[:, 0].copy(), arr[:, 1].copy()
            if flip_y:
                y = -y
            if flip_x:
                x = -x
            return x, y
        return None, None

    def axis_labels(self, view: str) -> tuple[str, str]:
        return _AXIS_LABELS.get(view.lower(), ('Dim 1', 'Dim 2'))

    # ------------------------------------------------------------------
    # Metadata color columns
    # ------------------------------------------------------------------

    # Categorical columns with more unique values than this are treated as
    # high-cardinality and downgraded to numeric (or skipped if non-numeric).
    _MAX_CATEGORIES = 500

    def color_columns(self) -> list[dict]:
        result = []
        for col in self.obs.columns:
            if col in self._skip_cols:
                continue
            vals = self.obs[col]
            n_unique = int(vals.nunique())
            is_cat = (
                (isinstance(vals.dtype, pd.CategoricalDtype) or vals.dtype == object)
                and n_unique <= self._MAX_CATEGORIES
            )
            tag = f'cat · {n_unique}' if is_cat else 'num'
            result.append({
                'label': f'{col}  [{tag}]',
                'value': col,
                'is_categorical': is_cat,
                'n_unique': n_unique,
            })
        result.sort(key=lambda x: (not x['is_categorical'], x['value'].lower()))
        return result

    def get_color_info(self, col: str):
        """Returns (color_vals, is_categorical, cat_colors_or_None)."""
        if col not in self.obs.columns:
            return None, False, None

        vals = self.obs[col]
        n_unique = int(vals.nunique())
        is_cat = (
            (isinstance(vals.dtype, pd.CategoricalDtype) or vals.dtype == object)
            and n_unique <= self._MAX_CATEGORIES
        )

        if is_cat:
            if hasattr(vals, 'cat'):
                categories = list(vals.cat.categories)
            else:
                categories = sorted({v for v in vals if pd.notna(v)}, key=str)

            if col in self.uns_colors:
                palette = list(self.uns_colors[col])
                while len(palette) < len(categories):
                    palette = palette * 2
                colors = palette[:len(categories)]
            else:
                base = (
                    list(px.colors.qualitative.Plotly)
                    + list(px.colors.qualitative.D3)
                    + list(px.colors.qualitative.Set1) * 4
                )
                palette = base
                while len(palette) < len(categories):
                    palette = palette + base
                colors = palette[:len(categories)]

            # datashader requires hex colors, not css rgb() strings
            def _to_hex(c: str) -> str:
                m = re.match(r'rgb\((\d+),\s*(\d+),\s*(\d+)\)', c)
                if m:
                    return '#{:02x}{:02x}{:02x}'.format(int(m.group(1)), int(m.group(2)), int(m.group(3)))
                return c
            colors = [_to_hex(c) for c in colors]

            return vals, True, list(zip(categories, colors))

        else:
            try:
                num = pd.to_numeric(vals, errors='coerce').values.astype(float)
            except Exception:
                num = np.zeros(len(vals))
            return num, False, None

    # ------------------------------------------------------------------
    # Gene expression
    # ------------------------------------------------------------------

    def get_gene_expr(self, gene_name: str) -> Optional[np.ndarray]:
        """Return expression values for one gene (lazy-loaded, cached)."""
        if gene_name not in self._gene_idx_map:
            return None
        if gene_name in self._gene_cache:
            return self._gene_cache[gene_name]

        gene_idx = self._gene_idx_map[gene_name]

        # If the CSC matrix is already built, use it directly (no file I/O)
        if self._csc_matrix is not None:
            col = self._csc_matrix.getcol(gene_idx)
            expr_full = np.asarray(col.todense()).flatten()
            expr = expr_full if self._idx is None else expr_full[self._idx]
            self._evict_gene_cache()
            self._gene_cache[gene_name] = expr.astype(float)
            return self._gene_cache[gene_name]

        with h5py.File(self.path, 'r') as f:
            X = f['X']
            if isinstance(X, h5py.Dataset):
                # Dense matrix: efficient column read
                expr_full = X[:, gene_idx].astype(float)
                expr = expr_full if self._idx is None else expr_full[self._idx]

            elif isinstance(X, h5py.Group) and 'data' in X:
                # Sparse CSR: load entire matrix, convert to CSC, cache both
                import scipy.sparse as sp
                data    = X['data'][:]
                indices = X['indices'][:]
                indptr  = X['indptr'][:]
                csr = sp.csr_matrix(
                    (data, indices, indptr),
                    shape=(self.n_cells_total, len(self.gene_names)),
                )
                self._csr_matrix = csr                    # row-efficient (DGE)
                self._csc_matrix = csr.tocsc()            # column-efficient (per-gene)
                col = self._csc_matrix.getcol(gene_idx)
                expr_full = np.asarray(col.todense()).flatten()
                expr = expr_full if self._idx is None else expr_full[self._idx]

            else:
                return None

        self._evict_gene_cache()
        self._gene_cache[gene_name] = expr.astype(float)
        return self._gene_cache[gene_name]

    def _evict_gene_cache(self) -> None:
        """Evict oldest cached gene if cache is at capacity."""
        if len(self._gene_cache) >= _GENE_CACHE_MAX:
            oldest = next(iter(self._gene_cache))
            del self._gene_cache[oldest]

    # ------------------------------------------------------------------
    # Differential gene expression
    # ------------------------------------------------------------------

    def _ensure_matrices(self) -> None:
        """Load and cache CSR/CSC matrices if not already done."""
        if self._csr_matrix is not None:
            return
        if self._csc_matrix is not None:
            self._csr_matrix = self._csc_matrix.tocsr()
            return
        import scipy.sparse as sp
        with h5py.File(self.path, 'r') as f:
            X = f['X']
            if isinstance(X, h5py.Group) and 'data' in X:
                csr = sp.csr_matrix(
                    (X['data'][:], X['indices'][:], X['indptr'][:]),
                    shape=(self.n_cells_total, len(self.gene_names)),
                )
                self._csr_matrix = csr
                self._csc_matrix = csr.tocsc()
            elif isinstance(X, h5py.Dataset):
                csr = sp.csr_matrix(X[:].astype(float))
                self._csr_matrix = csr
                self._csc_matrix = csr.tocsc()

    def run_dge(
        self,
        group1_point_indices: np.ndarray,
        group2_point_indices: np.ndarray,
        test: str = 'ttest',
    ) -> pd.DataFrame:
        """Differential gene expression + BH correction.

        Parameters
        ----------
        group1_point_indices, group2_point_indices :
            Integer indices into the *displayed/subsampled* scatter data
            (i.e., ``pointIndex`` values from Plotly ``selectedData``).
        test : 'ttest' (Welch, vectorised) or 'wilcoxon' (Mann-Whitney U).

        Returns
        -------
        pd.DataFrame sorted by |log2fc| desc with columns:
            gene, log2fc, mean1, mean2, pval, padj
        """
        if len(self.gene_names) == 0:
            return pd.DataFrame(columns=['gene', 'log2fc', 'mean1', 'mean2', 'pval', 'padj'])

        # Map displayed point indices → original h5ad row indices
        if self._idx is not None:
            orig_1 = self._idx[group1_point_indices]
            orig_2 = self._idx[group2_point_indices]
        else:
            orig_1 = group1_point_indices
            orig_2 = group2_point_indices

        self._ensure_matrices()

        if self._csr_matrix is None:
            return pd.DataFrame(columns=['gene', 'log2fc', 'mean1', 'mean2', 'pval', 'padj'])

        # Dense sub-matrices  (n_cells × n_genes) — manageable for 480 genes
        X1 = np.asarray(self._csr_matrix[orig_1, :].todense(), dtype=float)
        X2 = np.asarray(self._csr_matrix[orig_2, :].todense(), dtype=float)

        mean1 = X1.mean(axis=0).ravel()   # (n_genes,)
        mean2 = X2.mean(axis=0).ravel()

        # Log2 fold change with pseudo-count to handle zeros
        log2fc = np.log2(mean1 + 1) - np.log2(mean2 + 1)

        if test == 'wilcoxon':
            from scipy.stats import mannwhitneyu
            pvals = np.array([
                mannwhitneyu(X1[:, g], X2[:, g], alternative='two-sided').pvalue
                for g in range(X1.shape[1])
            ])
            pvals = np.nan_to_num(pvals, nan=1.0)
        else:
            from scipy.stats import ttest_ind
            _, pvals = ttest_ind(X1, X2, axis=0, equal_var=False)
            pvals = np.nan_to_num(np.asarray(pvals, dtype=float), nan=1.0)

        padj = _bh_correction(pvals)

        return (
            pd.DataFrame({
                'gene':   self.gene_names,
                'log2fc': log2fc,
                'mean1':  mean1,
                'mean2':  mean2,
                'pval':   pvals,
                'padj':   padj,
            })
            .assign(_absfc=lambda d: d['log2fc'].abs())
            .sort_values('_absfc', ascending=False)
            .drop(columns='_absfc')
            .reset_index(drop=True)
        )
