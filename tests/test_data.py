"""Tests for spatialxgene.data — the core data layer."""

import h5py
import numpy as np
import pandas as pd
import pytest

from spatialxgene.data import _bh_correction, _decode_bytes, SpatialData


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

class TestBHCorrection:
    def test_single_pvalue(self):
        pvals = np.array([0.05])
        padj = _bh_correction(pvals)
        assert padj[0] == pytest.approx(0.05)

    def test_known_values(self):
        # sorted p-values: rank-based adjustment then cummin from right
        pvals = np.array([0.01, 0.05, 0.10])
        padj = _bh_correction(pvals)
        expected = np.array([0.03, 0.075, 0.10])
        assert padj == pytest.approx(expected)

    def test_all_values_clamped_to_one(self):
        pvals = np.array([0.9, 0.95, 1.0])
        padj = _bh_correction(pvals)
        assert (padj <= 1.0).all()

    def test_empty_array(self):
        pvals = np.array([], dtype=float)
        padj = _bh_correction(pvals)
        assert len(padj) == 0

    def test_output_length_matches_input(self):
        pvals = np.random.default_rng(0).random(50)
        padj = _bh_correction(pvals)
        assert len(padj) == len(pvals)

    def test_monotone_after_correction(self):
        # After BH, padj values (when sorted by original rank) must be non-decreasing
        rng = np.random.default_rng(1)
        pvals = np.sort(rng.random(20))
        padj = _bh_correction(pvals)
        assert (np.diff(padj) >= -1e-12).all()


class TestDecodeBytes:
    def test_decodes_bytes(self):
        arr = np.array([b"foo", b"bar"])
        result = _decode_bytes(arr)
        assert result == ["foo", "bar"]

    def test_passes_through_strings(self):
        arr = np.array(["foo", "bar"], dtype=object)
        result = _decode_bytes(arr)
        assert result == ["foo", "bar"]

    def test_mixed(self):
        arr = [b"foo", "bar"]
        result = _decode_bytes(arr)
        assert result == ["foo", "bar"]


# ---------------------------------------------------------------------------
# SpatialData loading
# ---------------------------------------------------------------------------

class TestSpatialDataLoad:
    def test_n_cells(self, data):
        assert data.n_cells == 100

    def test_n_cells_total(self, data):
        assert data.n_cells_total == 100

    def test_name(self, data):
        assert data.name == "test"

    def test_gene_names(self, data):
        assert len(data.gene_names) == 10
        assert data.gene_names[0] == "gene_0"

    def test_obs_columns(self, data):
        assert "leiden" in data.obs.columns
        assert "score" in data.obs.columns

    def test_obs_index_length(self, data):
        assert len(data.obs) == 100


class TestSubsample:
    def test_subsample_count(self, h5ad_path):
        sd = SpatialData(h5ad_path, subsample=40, seed=0)
        assert sd.n_cells == 40
        assert sd.n_cells_total == 100

    def test_subsample_larger_than_total(self, h5ad_path):
        sd = SpatialData(h5ad_path, subsample=200)
        assert sd.n_cells == 100  # capped at total


# ---------------------------------------------------------------------------
# Views / embeddings
# ---------------------------------------------------------------------------

class TestAvailableViews:
    def test_has_spatial_and_umap(self, data):
        views = data.available_views()
        values = [v["value"] for v in views]
        assert "spatial" in values
        assert "umap" in values

    def test_no_pca_or_scvi(self, data):
        values = [v["value"] for v in data.available_views()]
        assert "pca" not in values
        assert "scvi" not in values


class TestGetCoords:
    def test_spatial_returns_correct_shape(self, data):
        x, y = data.get_coords("spatial")
        assert x.shape == (100,)
        assert y.shape == (100,)

    def test_umap_returns_correct_shape(self, data):
        x, y = data.get_coords("umap")
        assert x.shape == (100,)
        assert y.shape == (100,)

    def test_unknown_view_returns_none(self, data):
        x, y = data.get_coords("nonexistent")
        assert x is None and y is None

    def test_flip_y_negates_y(self, data):
        _, y_normal = data.get_coords("spatial", flip_y=False)
        _, y_flipped = data.get_coords("spatial", flip_y=True)
        np.testing.assert_array_almost_equal(y_flipped, -y_normal)

    def test_flip_x_negates_x(self, data):
        x_normal, _ = data.get_coords("spatial", flip_x=False)
        x_flipped, _ = data.get_coords("spatial", flip_x=True)
        np.testing.assert_array_almost_equal(x_flipped, -x_normal)

    def test_case_insensitive(self, data):
        x1, y1 = data.get_coords("Spatial")
        x2, y2 = data.get_coords("spatial")
        np.testing.assert_array_equal(x1, x2)


class TestAxisLabels:
    def test_spatial_labels(self, data):
        assert data.axis_labels("spatial") == ("X (µm)", "Y (µm)")

    def test_umap_labels(self, data):
        assert data.axis_labels("umap") == ("UMAP 1", "UMAP 2")

    def test_unknown_returns_default(self, data):
        assert data.axis_labels("foo") == ("Dim 1", "Dim 2")


# ---------------------------------------------------------------------------
# Metadata columns
# ---------------------------------------------------------------------------

class TestColorColumns:
    def test_returns_list_of_dicts(self, data):
        cols = data.color_columns()
        assert isinstance(cols, list)
        assert all(isinstance(c, dict) for c in cols)

    def test_leiden_is_categorical(self, data):
        cols = {c["value"]: c for c in data.color_columns()}
        assert "leiden" in cols
        assert cols["leiden"]["is_categorical"] is True
        assert cols["leiden"]["n_unique"] == 3

    def test_score_is_numerical(self, data):
        cols = {c["value"]: c for c in data.color_columns()}
        assert "score" in cols
        assert cols["score"]["is_categorical"] is False

    def test_skip_columns(self, h5ad_path):
        sd = SpatialData(h5ad_path, skip_columns={"score"})
        values = [c["value"] for c in sd.color_columns()]
        assert "score" not in values
        assert "leiden" in values


class TestGetColorInfo:
    def test_categorical_returns_cat_colors(self, data):
        vals, is_cat, cat_colors = data.get_color_info("leiden")
        assert is_cat is True
        assert cat_colors is not None
        assert len(cat_colors) == 3  # 3 leiden categories
        # each entry is (category, color_hex)
        cats, colors = zip(*cat_colors)
        assert all(c.startswith("#") for c in colors)

    def test_numerical_returns_array(self, data):
        vals, is_cat, cat_colors = data.get_color_info("score")
        assert is_cat is False
        assert cat_colors is None
        assert len(vals) == 100

    def test_unknown_column_returns_none(self, data):
        vals, is_cat, cat_colors = data.get_color_info("nonexistent_col")
        assert vals is None
        assert is_cat is False
        assert cat_colors is None

    def test_uns_colors_used_for_leiden(self, data):
        _, _, cat_colors = data.get_color_info("leiden")
        colors = [c for _, c in cat_colors]
        assert "#1f77b4" in colors  # from uns/leiden_colors in fixture


# ---------------------------------------------------------------------------
# Gene expression
# ---------------------------------------------------------------------------

class TestGetGeneExpr:
    def test_returns_array_of_correct_length(self, data):
        expr = data.get_gene_expr("gene_0")
        assert expr is not None
        assert len(expr) == 100

    def test_values_are_non_negative(self, data):
        expr = data.get_gene_expr("gene_0")
        assert (expr >= 0).all()

    def test_unknown_gene_returns_none(self, data):
        assert data.get_gene_expr("FAKE_GENE") is None

    def test_cached_on_second_access(self, data):
        expr1 = data.get_gene_expr("gene_1")
        expr2 = data.get_gene_expr("gene_1")
        assert expr1 is expr2  # same object from cache

    def test_subsampled_gene_expr_length(self, h5ad_path):
        sd = SpatialData(h5ad_path, subsample=40)
        expr = sd.get_gene_expr("gene_0")
        assert len(expr) == 40


# ---------------------------------------------------------------------------
# Differential gene expression
# ---------------------------------------------------------------------------

class TestRunDGE:
    def test_ttest_returns_correct_columns(self, data):
        g1 = np.arange(30)
        g2 = np.arange(60, 90)
        result = data.run_dge(g1, g2, test="ttest")
        assert list(result.columns) == ["gene", "log2fc", "mean1", "mean2", "pval", "padj"]

    def test_ttest_correct_number_of_genes(self, data):
        result = data.run_dge(np.arange(30), np.arange(60, 90), test="ttest")
        assert len(result) == 10

    def test_wilcoxon_returns_correct_columns(self, data):
        g1 = np.arange(30)
        g2 = np.arange(60, 90)
        result = data.run_dge(g1, g2, test="wilcoxon")
        assert list(result.columns) == ["gene", "log2fc", "mean1", "mean2", "pval", "padj"]

    def test_padj_between_0_and_1(self, data):
        result = data.run_dge(np.arange(30), np.arange(60, 90))
        assert result["padj"].between(0, 1).all()

    def test_pval_between_0_and_1(self, data):
        result = data.run_dge(np.arange(30), np.arange(60, 90))
        assert result["pval"].between(0, 1).all()

    def test_sorted_by_abs_log2fc(self, data):
        result = data.run_dge(np.arange(30), np.arange(60, 90))
        abs_fc = result["log2fc"].abs().values
        assert (np.diff(abs_fc) <= 1e-12).all()

    def test_means_are_non_negative(self, data):
        result = data.run_dge(np.arange(30), np.arange(60, 90))
        assert (result["mean1"] >= 0).all()
        assert (result["mean2"] >= 0).all()

    def test_no_genes_returns_empty_df(self, tmp_path):
        # h5ad with obs but no var/X
        path = tmp_path / "nogenes.h5ad"
        with h5py.File(path, "w") as f:
            obs = f.create_group("obs")
            obs.create_dataset("_index", data=np.array([b"c0", b"c1"]))
            obsm = f.create_group("obsm")
            obsm.create_dataset("spatial", data=np.zeros((2, 2), dtype=np.float32))
        sd = SpatialData(str(path))
        result = sd.run_dge(np.array([0]), np.array([1]))
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0
