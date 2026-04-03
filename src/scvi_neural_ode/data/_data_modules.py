import warnings
from math import ceil, floor
from typing import Optional

import numpy as np
import pandas as pd
import pytorch_lightning as pl
from anndata import AnnData

from scvi_neural_ode.data._loaders import AnnDataLoader, ConcatAnnDataLoader


def validate_data_split(
    n_samples: int,
    train_size: float,
    validation_size: Optional[float] = None,
    batch_size: Optional[int] = None,
    drop_last: bool = False,
):
    """
    Check data splitting parameters and return n_train and n_val.

    Parameters
    ----------
    n_samples
        Number of samples to split
    train_size
        Size of train set. Need to be: 0 < train_size <= 1.
    validation_size
        Size of validation set. Need to be 0 <= validation_size < 1
    """
    if train_size > 1.0 or train_size <= 0.0:
        raise ValueError("Invalid train_size. Must be: 0 < train_size <= 1")

    n_train = ceil(train_size * n_samples)

    assert n_train, (
        f"With n_samples={n_samples}, train_size={train_size} and validation_size={validation_size}, the train set "
        "will be empty. Adjust any of these parameters to obtain a non-empty train set. "
    )

    if validation_size is None:
        n_val = n_samples - n_train
    elif validation_size >= 1.0 or validation_size < 0.0:
        raise ValueError("Invalid validation_size. Must be 0 <= validation_size < 1")
    elif (train_size + validation_size) > 1:
        raise ValueError("train_size + validation_size must be between 0 and 1")
    else:
        n_val = floor(n_samples * validation_size)

    if batch_size is not None:
        remainder_batch = n_train % batch_size
        if 0 < remainder_batch < 3 and not drop_last:
            warnings.warn(
                f"Last batch will have a small size of {remainder_batch}. Consider changing `batch_size` to avoid errors during model training.",
                UserWarning,
                stacklevel=2,
            )
        # else:
        #     n_train -= remainder_batch
        #     if n_val:
        #         n_val += remainder_batch
        #         warnings.warn(
        #             f"{remainder_batch} cells moved from train set to validation set.",
        #             UserWarning,
        #             stacklevel=2
        #         )

    return n_train, n_val


class LightningAnnDataModule(pl.LightningDataModule):
    """
    Creates data loaders for train, validation and test sets.

    If train_size + validation_size < 1 then test_set is non-empty.

    Input
    -----
    adata
        AnnData object.
    train_size
        Size of train set, between 0 and 1 as proportion of total number of observations. Default is 0.9
    validation_size
        Size of validation set.
    shuffle_set_split
        Whether to shuffle indices before splitting. If False then train, val and tests sets are split in the
        sequential order in the data according to the `train_size` and `validation_size` proportions.
    shuffle_train_batches
        Whether to shuffle indices in a train data loader.
    _data_loader_class
        Class of scskeleton data loader.
    **data_loader_kwargs
        Keyword argument for the data loaders.
    """

    def __init__(
        self,
        adata: AnnData,
        train_size: float = 0.9,
        validation_size: Optional[float] = None,
        shuffle_set_split: bool = True,
        shuffle_train_batches: bool = True,
        seed: int = 0,
        _data_loader_class=AnnDataLoader,
        **data_loader_kwargs,
    ):
        # TO DO: add gpu option.
        super().__init__()
        self.adata = adata
        self.train_size = float(train_size)
        self.validation_size = validation_size
        self.shuffle_set_split = shuffle_set_split
        self.shuffle_train_batches = shuffle_train_batches
        self.data_loader_kwargs = data_loader_kwargs
        self._data_loader_class = _data_loader_class
        self.drop_last = data_loader_kwargs.pop("drop_last", False)
        self.seed = seed

        assert (
            "shuffle" not in data_loader_kwargs.keys()
        ), "Use `shuffle_train_batches` instead of `shuffle`."

        if "indices_in_every_batch" in data_loader_kwargs.keys():
            self.indices_in_every_batch = data_loader_kwargs.pop("indices_in_every_batch")
        else:
            self.indices_in_every_batch = None

        self.n_train, self.n_val = validate_data_split(
            n_samples=self.adata.n_obs,
            train_size=train_size,
            validation_size=validation_size,
            # Default batch_size for data loader is 128, but it's assigned to an attribute anywhere.
            batch_size=self.data_loader_kwargs.get("batch_size", None),
            drop_last=self.drop_last,
        )

    def setup(self, stage: Optional[str] = None):
        # TO DO: add seed option?
        indices = np.arange(self.adata.n_obs)
        # Remove indices_in_every_batch from indices.
        if self.indices_in_every_batch is not None:
            indices = np.setdiff1d(indices, self.indices_in_every_batch)
        if self.shuffle_set_split:
            random_state = np.random.RandomState(seed=self.seed)  # change this
            indices = random_state.permutation(indices)
        self.train_idx = indices[: self.n_train]
        self.val_idx = indices[self.n_train : (self.n_train + self.n_val)]
        self.test_idx = indices[(self.n_train + self.n_val) :]
        # None of these 3 need to contain indices_in_every_batch, as that will be added by the train dataloader.

    def train_dataloader(self):
        return self._data_loader_class(
            self.adata,
            indices=self.train_idx,
            indices_in_every_batch=self.indices_in_every_batch,
            shuffle=self.shuffle_train_batches,
            drop_last=self.drop_last,
            **self.data_loader_kwargs,
        )

    def val_dataloader(self):
        if len(self.val_idx) > 0:
            return self._data_loader_class(
                self.adata,
                indices=self.val_idx,
                shuffle=False,
                indices_in_every_batch=None,
                drop_last=False,
                **self.data_loader_kwargs,
            )
        else:
            return []

    def test_dataloader(self):
        if len(self.test_idx) > 0:
            return self._data_loader_class(
                self.adata,
                indices=self.test_idx,
                shuffle=False,
                indices_in_every_batch=None,
                drop_last=False,
                **self.data_loader_kwargs,
            )
        else:
            return []


class ConditionalAnnDataModule(pl.LightningDataModule):
    """
    Creates concat data loaders for train, validation and test sets.
    """

    def __init__(
        self,
        adata: AnnData,
        key: str,
        train_size: float = 0.9,
        validation_size: Optional[float] = None,
        shuffle_train_batches: bool = False,
        seed: int = 0,
        **data_loader_kwargs,
    ):
        super().__init__()
        self.adata = adata
        self.train_size = float(train_size)
        self.validation_size = validation_size
        self.shuffle_train_batches = shuffle_train_batches
        self.data_loader_kwargs = data_loader_kwargs
        self.drop_last = data_loader_kwargs.pop("drop_last", False)
        self.indices_in_every_batch = None
        self.seed = seed

        assert (
            "shuffle" not in data_loader_kwargs.keys()
        ), "Use `shuffle_train_batches` instead of `shuffle`."

        self.key_value_list = pd.Categorical(adata.obs[key]).categories.values
        self.indices_per_val = []
        for val in self.key_value_list:
            self.indices_per_val += [np.where(adata.obs_vector(key) == val)[0]]

    def setup(self, stage: Optional[str] = None):
        self.train_idx, self.val_idx, self.test_idx = [], [], []
        for i in range(len(self.indices_per_val)):
            len_idx = len(self.indices_per_val[i])
            n_train, n_validation = validate_data_split(
                n_samples=len_idx,
                train_size=self.train_size,
                validation_size=self.validation_size,
                batch_size=self.data_loader_kwargs.get("batch_size", None),
                drop_last=self.drop_last,
            )
            rs = np.random.RandomState(seed=self.seed)
            permutation = rs.choice(self.indices_per_val[i], size=len_idx, replace=False)
            self.val_idx += [permutation[:n_validation].astype(int)]
            self.train_idx += [permutation[n_validation : (n_train + n_validation)].astype(int)]
            self.test_idx += [permutation[(n_train + n_validation) :].astype(int)]

    def train_dataloader(self):
        return ConcatAnnDataLoader(
            adata=self.adata,
            indices_list=self.train_idx,
            shuffle=self.shuffle_train_batches,
            drop_last=self.drop_last,
            **self.data_loader_kwargs,
        )

    def val_dataloader(self):
        if len(self.val_idx):
            return ConcatAnnDataLoader(
                adata=self.adata,
                indices_list=self.val_idx,
                shuffle=False,
                drop_last=False,
                **self.data_loader_kwargs,
            )
        else:
            pass

    def test_dataloader(self):
        if len(self.test_idx):
            return ConcatAnnDataLoader(
                adata=self.adata,
                indices_list=self.test_idx,
                shuffle=False,
                drop_last=False,
                **self.data_loader_kwargs,
            )
        else:
            pass
