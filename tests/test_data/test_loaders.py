from math import ceil, floor

import numpy as np

from scvi_neural_ode import _CONSTANTS
from scvi_neural_ode.data import (
    AnnDataLoader,
    ConcatAnnDataLoader,
    LightningAnnDataModule,
    register_tensor_from_anndata,
)


def test_loader(adata):
    _indices = np.arange(adata.n_obs)
    adata.obs["_indices"] = _indices
    batch_size = 50
    register_tensor_from_anndata(
        adata, registry_key="_indices", adata_attr_name="obs", adata_key_name="_indices"
    )
    dl = AnnDataLoader(adata, shuffle=False, batch_size=batch_size)
    for j, tensors in enumerate(dl):
        x = tensors[_CONSTANTS.X_KEY]
        if j < adata.n_obs // batch_size:
            assert x.shape == (batch_size, adata.n_vars)
        else:
            assert x.shape == (adata.n_obs % batch_size, adata.n_vars)  # last batch
        ind_x = tensors["_indices"].squeeze()
        np.testing.assert_array_equal(
            _indices[batch_size * j : batch_size * (j + 1)], ind_x
        )  # No shuffle

    ind_query = [1, 4, 6, 7, 8]
    dl = AnnDataLoader(adata, indices=ind_query)
    # Note len(indices) < batch_size
    for tensors in dl:
        ind_x = tensors["_indices"].squeeze()
        np.testing.assert_array_equal(ind_x, ind_query)


def test_concat_loader(adata):
    indices1 = np.arange(50)
    indices2 = np.arange(100)
    indices_list = [indices1, indices2]
    batch_size = 22
    adata.obs["_indices"] = np.arange(adata.n_obs)
    register_tensor_from_anndata(
        adata, registry_key="_indices", adata_attr_name="obs", adata_key_name="_indices"
    )
    dl = ConcatAnnDataLoader(
        adata=adata, indices_list=indices_list, shuffle=False, batch_size=batch_size
    )
    L1, L2 = len(indices1) // batch_size + 1, len(indices2) // batch_size + 1
    assert len(dl) == max(L1, L2)
    for i, tensors in enumerate(dl):
        assert len(tensors) == 2  # 2 lists of indices
        x1 = tensors[0][_CONSTANTS.X_KEY]
        x2 = tensors[1][_CONSTANTS.X_KEY]
        if i == L1 - 1:
            assert x1.shape == (len(indices1) % batch_size, adata.n_vars)
        elif i == L2 - 1:
            assert x2.shape == (len(indices2) % batch_size, adata.n_vars)
        else:
            assert x1.shape == (batch_size, adata.n_vars)
            assert x2.shape == (batch_size, adata.n_vars)


def test_split(adata):
    dm = LightningAnnDataModule(adata, train_size=0.5, validation_size=0.3, shuffle_set_split=False)
    dm.setup()
    assert isinstance(dm.train_idx, np.ndarray)
    assert isinstance(dm.val_idx, np.ndarray)
    assert isinstance(dm.test_idx, np.ndarray)

    n_train = ceil(adata.n_obs * 0.5)
    n_val = floor(adata.n_obs * 0.3)
    n_test = adata.n_obs - n_train - n_val

    np.testing.assert_array_equal(dm.train_idx, np.arange(n_train))
    np.testing.assert_array_equal(dm.val_idx, np.arange(n_train, n_train + n_val))
    np.testing.assert_array_equal(
        dm.test_idx,
        np.arange(n_val + n_train, n_val + n_train + n_test),
    )


def test_indices_in_every_batch(adata):
    adata.obs["_indices"] = np.arange(adata.n_obs)
    register_tensor_from_anndata(
        adata, registry_key="_indices", adata_attr_name="obs", adata_key_name="_indices"
    )
    idx_to_repeat = 1
    dm = LightningAnnDataModule(
        adata,
        train_size=0.5,
        validation_size=0.3,
        indices_in_every_batch=[idx_to_repeat],
    )
    dm.setup()

    val_tensors = [tensors for tensors in dm.val_dataloader()][0]
    assert idx_to_repeat not in val_tensors["_indices"].squeeze().numpy()

    test_tensors = [tensors for tensors in dm.test_dataloader()][0]
    assert idx_to_repeat not in test_tensors["_indices"].squeeze().numpy()

    train_dl = dm.train_dataloader()
    for tensors in train_dl:
        ind_x = tensors["_indices"].squeeze()
        assert ind_x[0].item() == idx_to_repeat


def test_shuffle_train_batches(adata):
    adata.obs["_indices"] = np.arange(adata.n_obs)
    register_tensor_from_anndata(
        adata, registry_key="_indices", adata_attr_name="obs", adata_key_name="_indices"
    )

    batch_size = 50
    train_size = 0.9
    dm = LightningAnnDataModule(
        adata,
        train_size=train_size,
        batch_size=batch_size,
        shuffle_set_split=True,
        indices_in_every_batch=None,
        shuffle_train_batches=False,
    )
    dm.setup()
    train_dl = dm.train_dataloader()

    indices = np.arange(adata.n_obs)
    random_state = np.random.RandomState(seed=dm.seed)
    indices = random_state.permutation(indices)
    n_train = ceil(adata.n_obs * train_size)
    train_idx = indices[:n_train]  # or can just call dm.train_idx

    for i, tensors in enumerate(train_dl):
        ind_x = tensors["_indices"].squeeze().numpy()
        train_idx_batch = train_idx[i * batch_size : (i + 1) * batch_size]
        np.testing.assert_array_equal(ind_x, train_idx_batch)

    indices_in_every_batch = [10, 20, 30]
    n_recurring_indices = len(indices_in_every_batch)
    dm = LightningAnnDataModule(
        adata,
        train_size=train_size,
        batch_size=batch_size,
        shuffle_set_split=True,
        indices_in_every_batch=indices_in_every_batch,
        shuffle_train_batches=False,
    )
    dm.setup()
    train_dl = dm.train_dataloader()
    train_idx = dm.train_idx
    for i, tensors in enumerate(train_dl):
        ind_x = tensors["_indices"].squeeze().numpy()
        # Test first entries are the recurring indices.
        np.testing.assert_array_equal(ind_x[:n_recurring_indices], indices_in_every_batch)

        # Test the rest of entries in the batch are in the expected order.
        train_idx_batch = train_idx[i * batch_size : (i + 1) * batch_size]
        np.testing.assert_array_equal(ind_x[n_recurring_indices:], train_idx_batch)
