from copy import copy
from itertools import cycle
from typing import List, Optional, Union

import numpy as np
from anndata import AnnData
from torch.utils.data import DataLoader

from scvi_neural_ode import _CONSTANTS
from scvi_neural_ode.data._anndataset import AnnDataSet
from scvi_neural_ode.data._sampler import BatchSampler


class AnnDataLoader(DataLoader):
    """
    DataLoader for loading tensors from AnnData objects.

    Adapted from scvi-tools: https://github.com/scverse/scvi-tools/blob/main/src/scvi/dataloaders/_ann_dataloader.py

    Input
    -----
    adata
        AnnData object.
    shuffle
        Whether to shuffle indices before sampling.
    indices
        Indices of observations in adata to load.
    batch_size
        Minibatch size to load per iteration.
    getitem_tensors
        Dictionary of keys in data registry and values equal to desired numpy loading type. If None, uses all keys in
        data registry.
    drop_last
        If dataset size is not divisible by `batch_size` then drop the last incomplete batch. If False and non-divisibility
        then the last batch will be smaller than `batch_size`.
    _anndataset_class
        Class of scskeleton AnnDataSet
    indices_in_every_batch
        Whether to include a set of indices at the beginning of each batch.
    data_loader_kwargs
        Keyword arguments for class torch.utils.data.DataLoader
    """

    def __init__(
        self,
        adata: AnnData,
        shuffle: bool = False,
        indices: Optional[np.ndarray] = None,
        batch_size: int = 128,
        getitem_tensors: Optional[dict] = None,
        drop_last: bool = False,
        _anndataset_class=AnnDataSet,
        indices_in_every_batch: Optional[Union[list, int]] = None,
        **data_loader_kwargs,
    ):
        if getitem_tensors is not None:
            data_registry = adata.uns["data_registry"]
            for key in getitem_tensors:
                if key not in data_registry:
                    raise KeyError("{} not included in data registry".format(key))
        self.dataset = _anndataset_class(adata, getitem_tensors=getitem_tensors)
        self._anndataset_class = _anndataset_class
        if isinstance(indices_in_every_batch, int):
            indices_in_every_batch = [indices_in_every_batch]

        sampler_kwargs = {
            "batch_size": batch_size,
            "shuffle": shuffle,
            "drop_last": drop_last,
            "indices_in_every_batch": indices_in_every_batch,
        }
        if indices is None:
            indices = np.arange(adata.shape[0])
        else:
            indices = np.asarray(indices)
        sampler_kwargs["indices"] = indices

        self.indices = indices
        self.sampler_kwargs = sampler_kwargs
        sampler = BatchSampler(**self.sampler_kwargs)
        self.data_loader_kwargs = copy(data_loader_kwargs)  # why copy?
        self.data_loader_kwargs.update({"sampler": sampler, "batch_size": None})

        super().__init__(self.dataset, **self.data_loader_kwargs)


class ConcatAnnDataLoader(DataLoader):
    def __init__(
        self,
        adata: Optional[AnnData] = None,
        dataloaders: Optional[List] = None,
        indices_list: Optional[List[List[int]]] = None,
        shuffle: bool = False,
        batch_size: int = 128,
        getitem_tensors: Optional[dict] = None,
        drop_last: bool = False,
        **data_loader_kwargs,
    ):
        # Check you can still use this for JointVI!
        assert not (adata is not None and dataloaders is not None)
        self.data_loader_kwargs = data_loader_kwargs
        self.getitem_tensors = getitem_tensors
        self._shuffle = shuffle
        self._batch_size = batch_size
        self._drop_last = drop_last

        if dataloaders is None and adata is not None:
            assert indices_list is not None, "ConcatAnnDataLoader requires indices_list input"
            self.dataloaders = []
            self.adata = adata
            for indices in indices_list:
                self.dataloaders.append(
                    AnnDataLoader(
                        adata,
                        indices=indices,
                        shuffle=shuffle,
                        batch_size=batch_size,
                        getitem_tensors=getitem_tensors,
                        drop_last=drop_last,
                        **data_loader_kwargs,
                    )
                )
        elif dataloaders is not None and adata is None:
            self.dataloaders = dataloaders
        else:
            raise ValueError("Provide either adata or a list of dataloaders.")

        self.largest_dl_idx = np.argmax([len(dl) for dl in self.dataloaders])
        self.largest_dl = self.dataloaders[self.largest_dl_idx]
        super().__init__(self.largest_dl, **data_loader_kwargs)

    def __len__(self):
        return len(self.largest_dl)

    def __iter__(self):
        """
        Iter method for concat data loader.
        Will iter over the dataloader with the most data while cycling through
        the data in the other dataloaders. The order of data in returned iter_list
        is the same as indices_list.
        """
        iter_list = [
            cycle(dl) if i != self.largest_dl_idx else dl for i, dl in enumerate(self.dataloaders)
        ]
        return zip(*iter_list)

