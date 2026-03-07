"""Shared pytest fixtures — builds a minimal synthetic .h5ad file once per session."""

import numpy as np
import h5py
import scipy.sparse as sp
import pytest

from spatialxgene.data import SpatialData


@pytest.fixture(scope="session")
def h5ad_path(tmp_path_factory):
    """Create a minimal valid h5ad file with 100 cells and 10 genes."""
    tmp = tmp_path_factory.mktemp("data")
    path = tmp / "test.h5ad"

    n_cells = 100
    n_genes = 10
    rng = np.random.default_rng(0)

    # Sparse CSR matrix (30% density)
    dense = rng.random((n_cells, n_genes)).astype(np.float32)
    dense[rng.random((n_cells, n_genes)) < 0.7] = 0.0
    csr = sp.csr_matrix(dense)

    with h5py.File(path, "w") as f:
        # obs
        obs = f.create_group("obs")
        obs.create_dataset("_index", data=np.array([f"cell_{i}".encode() for i in range(n_cells)]))

        # categorical: leiden (3 categories)
        leiden = obs.create_group("leiden")
        leiden.create_dataset("codes", data=np.array([i % 3 for i in range(n_cells)], dtype=np.int8))
        leiden.create_dataset("categories", data=np.array([b"0", b"1", b"2"]))

        # numerical: score
        obs.create_dataset("score", data=rng.random(n_cells).astype(np.float32))

        # obsm
        obsm = f.create_group("obsm")
        obsm.create_dataset("spatial", data=rng.random((n_cells, 2)).astype(np.float32) * 1000)
        obsm.create_dataset("X_umap",  data=rng.random((n_cells, 2)).astype(np.float32))

        # var
        var = f.create_group("var")
        var.create_dataset("_index", data=np.array([f"gene_{i}".encode() for i in range(n_genes)]))

        # X (sparse CSR)
        X = f.create_group("X")
        X.create_dataset("data",    data=csr.data)
        X.create_dataset("indices", data=csr.indices)
        X.create_dataset("indptr",  data=csr.indptr)

        # uns: colors for leiden
        uns = f.create_group("uns")
        uns.create_dataset("leiden_colors", data=np.array([b"#1f77b4", b"#ff7f0e", b"#2ca02c"]))

    return str(path)


@pytest.fixture()
def data(h5ad_path):
    """Fresh SpatialData instance for each test."""
    return SpatialData(h5ad_path)
