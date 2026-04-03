import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

from scvi_neural_ode import _CONSTANTS
from scvi_neural_ode.data import (
    AnnDataSet,
    add_categorical_obs,
    get_from_registry,
    register_tensor_from_anndata,
    setup_data_registry,
    synthetic_iid,
    update_obs_colors,
)


def test_setup_data_registry():
    X = np.random.negative_binomial(5, 0.3, size=(50, 5))
    adata = AnnData(X)
    adata.obs["batch"] = np.random.choice(["a", "b", "c"], size=adata.n_obs)
    adata.obs["cat"] = np.random.choice(["A", "B", "C"], size=adata.n_obs)
    adata.obs["cont1"] = np.random.normal(size=adata.n_obs)
    adata.obs["cont2"] = np.random.normal(size=adata.n_obs)
    all_keys = [
        _CONSTANTS.X_KEY,
        _CONSTANTS.BATCH_KEY,
        _CONSTANTS.CAT_COV_KEY,
        _CONSTANTS.CONT_COV_KEY,
    ]
    setup_data_registry(adata, batch_key="batch", extra_cat_covs=["cat"], extra_cont_covs=["cont1"])
    data_registry = adata.uns["data_registry"]
    np.testing.assert_equal(all_keys, list(data_registry.keys()))
    np.testing.assert_equal(data_registry[_CONSTANTS.BATCH_KEY], ["obs", "batch"])
    np.testing.assert_equal(data_registry[_CONSTANTS.CAT_COV_KEY], ["obs", "cat"])
    np.testing.assert_equal(data_registry[_CONSTANTS.CONT_COV_KEY], ["obs", "cont1"])

    del adata.uns["data_registry"]
    setup_data_registry(adata, batch_key="batch", extra_cont_covs=["cont1", "cont2"])
    data_registry = adata.uns["data_registry"]
    assert _CONSTANTS.CAT_COV_KEY not in data_registry.keys()
    np.testing.assert_equal(data_registry[_CONSTANTS.CONT_COV_KEY], ["obs", ["cont1", "cont2"]])

    with pytest.raises(ValueError):
        setup_data_registry(adata, batch_key="batch", extra_cont_covs=["ABC"])


def test_anndataset_getitem(adata):
    # adata will have data registry set up.
    # pass tensors to get
    tensors_to_get = [_CONSTANTS.BATCH_KEY]
    bd = AnnDataSet(adata, getitem_tensors=tensors_to_get)
    np.testing.assert_array_equal(tensors_to_get, list(bd[0].keys()))
    # bd[0] just returns a row (see __getitem__)

    # pass dict of tensors and their associated types.
    bd = AnnDataSet(
        adata,
        getitem_tensors={_CONSTANTS.X_KEY: np.int64, _CONSTANTS.BATCH_KEY: np.float64},
    )
    assert isinstance(bd.keys_and_types, dict)
    assert bd[1][_CONSTANTS.X_KEY].dtype == np.int64
    assert bd[1][_CONSTANTS.BATCH_KEY].dtype == np.float64

    # check that by default we get all registered tensors with default types.
    bd = AnnDataSet(adata)
    all_registered_tensors = list(adata.uns["data_registry"].keys())
    np.testing.assert_array_equal(all_registered_tensors, list(bd[0].keys()))
    for value in bd[0].values():
        assert isinstance(value, np.ndarray)
        assert value.dtype == np.float32
    assert bd[:5][_CONSTANTS.X_KEY].shape == (5, adata.n_vars)
    assert bd[:5][_CONSTANTS.BATCH_KEY].shape == (5, 1)
    # Should the array shape (e.g. X) for 1 sample be (1, n_vars) or (n_vars,)?


def test_get_from_registry(adata):
    X = get_from_registry(adata, _CONSTANTS.X_KEY)
    assert X.shape == (adata.n_obs, adata.n_vars)
    batch = get_from_registry(adata, _CONSTANTS.BATCH_KEY)
    assert batch.shape == (adata.n_obs, 1)

    # Handle multiple keys per attribute.
    adata.uns["data_registry"].update({"dummy": ["obs", ["batch", "cont1"]]})
    data = get_from_registry(adata, "dummy")
    pd.testing.assert_frame_equal(data, adata.obs[["batch", "cont1"]])
    del adata.uns["data_registry"]["dummy"]


def test_register_tensor_from_anndata(adata):
    register_tensor_from_anndata(
        adata, registry_key="test", adata_attr_name="obs", adata_key_name="cont1"
    )
    assert "test" in adata.uns["data_registry"]
    assert adata.uns["data_registry"]["test"] == ["obs", "cont1"]
    # check that new tensor is returned by __getitem__
    bd = AnnDataSet(adata)
    all_registered_tensors = list(adata.uns["data_registry"].keys())
    np.testing.assert_array_equal(all_registered_tensors, list(bd[0].keys()))


def test_add_categorical_obs(adata):
    labels = np.random.choice(["a", "b", "c"], size=adata.n_obs)
    labels_df = pd.DataFrame(labels, index=adata.obs.index)
    # Test whether it still adds labels even when not all adata samples are in label_df.
    labels_df = labels_df[10:]
    add_categorical_obs(adata, labels_df, "random_labels")

    assert "random_labels" in adata.obs.columns
    np.testing.assert_array_equal(adata.obs["random_labels"].values[10:], labels[10:])
    assert adata.obs["random_labels"][:10].isna().all()


def test_edit_colors():
    adata = synthetic_iid(n_batches=5)
    uns_key = "batch_colors"
    adata.uns[uns_key] = ["a", "b", "c", "d", "e"]
    new_colors = {0: "A", 1: "B", 2: "C", 3: "D", 4: "E"}
    update_obs_colors(adata, obs_name="batch", new_colors_dict=new_colors)
    assert adata.uns[uns_key] == ["A", "B", "C", "D", "E"]
    new_colors = {0: "a", 2: "c", 4: "e"}
    update_obs_colors(adata, obs_name="batch", new_colors_dict=new_colors)
    assert adata.uns[uns_key] == ["a", "B", "c", "D", "e"]
