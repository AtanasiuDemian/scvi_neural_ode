import logging
from typing import Dict, List, Union

import numpy as np
import pandas as pd
from anndata import AnnData
from torch.utils.data import Dataset

from scvi_neural_ode.data._anndata import get_from_registry


class AnnDataSet(Dataset):
    def __init__(
        self,
        adata: AnnData,
        getitem_tensors: Union[List[str], Dict[str, type]] = None,
    ):
        self.adata = adata
        self.keys_and_types = None
        self.setup_getitem(getitem_tensors)  # sets up self.keys_and_types
        self.data = {
            key: get_from_registry(self.adata, key) for key, _ in self.keys_and_types.items()
        }

    @property
    def registered_keys(self):
        return self.adata.uns["data_registry"].keys()

    def setup_getitem(self, getitem_tensors: Union[List[str], Dict[str, type]] = None):
        """
        getitem_tensors: list/dictionary of keys in the data registry to return when calling __getitem__

        e.g. getitem_tensors = ['X', 'n_counts'] or {'X': np.int32, 'n_counts': np.int32}.

        Updates self.keys_and_types. If no getitem_tensors are provided, use all registered keys.
        """
        registered_keys = self.registered_keys
        if isinstance(getitem_tensors, List):
            keys = getitem_tensors
            keys_and_types = {key: np.float32 for key in keys}

        elif isinstance(getitem_tensors, Dict):
            keys = getitem_tensors.keys()
            keys_and_types = getitem_tensors

        elif getitem_tensors is None:
            keys = registered_keys
            keys_and_types = {key: np.float32 for key in keys}

        for key in keys:
            assert key in registered_keys, "{} not in registered keys".format(key)
        self.keys_and_types = keys_and_types

    def __getitem__(self, idx: List[int]):
        data_numpy = {}
        for key, dtype in self.keys_and_types.items():
            data = self.data[key]
            if isinstance(data, np.ndarray):
                data_numpy[key] = data[idx].astype(dtype)
            elif isinstance(data, pd.DataFrame):
                data_numpy[key] = data.iloc[idx, :].to_numpy().astype(dtype)
            else:
                data_numpy[key] = data[idx].toarray().astype(dtype)

        return data_numpy

    def __len__(self):
        return self.adata.shape[0]
