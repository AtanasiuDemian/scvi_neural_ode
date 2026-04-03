import inspect
import os
import pickle
from abc import ABC, abstractmethod
from typing import Optional, Sequence, Union

import numpy as np
import torch
from anndata import AnnData

from scvi_neural_ode.data import AnnDataLoader, LightningAnnDataModule

from ._utils import _initialize_model, _load_saved_files, _validate_var_names


class BaseModelAbstractClass(ABC):
    """
    Create AnnDataLoader object for data iteration.
    """

    def __init__(
        self,
        adata: Optional[AnnData] = None,
        batch_size: Optional[int] = 128,
    ):
        if adata is not None:
            self.adata = adata
            # add step to validate AnnData object.
        self.is_trained_ = False
        self.batch_size = batch_size
        self.train_indices_ = None
        self.test_indices_ = None
        self.validation_indices_ = None
        self._data_loader_cls = AnnDataLoader
        self._data_module_cls = LightningAnnDataModule

    def to_device(self, device: Union[str, int]):
        self.module.to(torch.device(device))

    @property
    def device(self):
        return self.module.device

    def _make_data_loader(
        self,
        adata: AnnData,
        indices: Optional[Sequence[int]] = None,
        batch_size: Optional[int] = None,
        shuffle: bool = False,
        loader_class=None,
        **data_loader_kwargs,
    ):
        adata = self.adata if adata is None and hasattr(self, "adata") else adata
        if batch_size is None:
            batch_size = self.batch_size
        if indices is None:
            indices = np.arange(adata.shape[0])
        if loader_class is None:
            loader_class = self._data_loader_cls
        dl = loader_class(
            adata,
            shuffle=shuffle,
            indices=indices,
            batch_size=batch_size,
            **data_loader_kwargs,
        )

        return dl

    def _validate_anndata(self, adata: Optional[AnnData] = None):
        pass

    @property
    def is_trained(self):
        return self.is_trained_

    @property
    def train_indices(self):
        return self.train_indices_

    @property
    def test_indices(self):
        return self.test_indices_

    @property
    def validation_indices(self):
        return self.validation_indices_

    @is_trained.setter
    def is_trained(self, value):
        self.is_trained_ = value

    @train_indices.setter
    def train_indices(self, value):
        self.train_indices_ = value

    @test_indices.setter
    def test_indices(self, value):
        self.test_indices_ = value

    @validation_indices.setter
    def validation_indices(self, value):
        self.validation_indices_ = value

    @abstractmethod
    def train(self):
        pass

    def _get_user_attributes(self):
        # Need to learn what the inspect module does.
        attributes = inspect.getmembers(self, lambda a: not (inspect.isroutine(a)))
        attributes = [a for a in attributes if not (a[0].startswith("__") and a[0].endswith("__"))]
        attributes = [a for a in attributes if not a[0].startswith("_abc_")]

        return attributes

    def _get_init_params(self, locals):
        init = self.__init__
        sig = inspect.signature(init)
        init_params = [p for p in sig.parameters]
        user_params = {p: locals[p] for p in locals if p in init_params}
        user_params = {k: v for (k, v) in user_params.items() if not isinstance(v, AnnData)}

        return user_params

    def save(
        self,
        dir_path: str,
        overwrite: bool = False,
        save_anndata: bool = False,
        **anndata_write_kwargs,
    ):
        """
        "Save the state of the model."
        Does not save trainer optimizer state or trainer history.
        """
        user_attributes = self._get_user_attributes()
        # save attributes with '_' at the end
        user_attributes = {a[0]: a[1] for a in user_attributes if a[0][-1] == "_"}
        # "save the model state dict and the trainer state dict only"
        if not os.path.exists(dir_path) or overwrite:
            os.makedirs(dir_path, exist_ok=overwrite)
        else:
            raise ValueError("{} already exists.".format(dir_path))

        if save_anndata:
            self.adata.write(os.path.join(dir_path, "adata.h5ad"), **anndata_write_kwargs)
        model_save_path = os.path.join(dir_path, "model_params.pt")
        attr_save_path = os.path.join(dir_path, "attr.pkl")
        var_names_save_path = os.path.join(dir_path, "var_names.csv")

        var_names = self.adata.var_names.astype(str)
        var_names = var_names.to_numpy()
        np.savetxt(var_names_save_path, var_names, fmt="%s")
        torch.save(self.module.state_dict(), model_save_path)
        with open(attr_save_path, "wb") as f:
            pickle.dump(user_attributes, f)

    @classmethod
    def load(
        cls,
        dir_path: str,
        adata: Optional[AnnData] = None,
    ):
        """
        Instantiate model from saved output.
        """
        # Need to do something about default AnnData argument.
        load_data = adata is None
        attr_dict, var_names, model_state_dict, new_anndata = _load_saved_files(dir_path, load_data)
        adata = new_anndata if new_anndata is not None else adata
        _validate_var_names(adata, var_names)
        # transfer_anndata_setup()
        model = _initialize_model(cls, adata, attr_dict)
        for attr, val in attr_dict.items():
            setattr(model, attr, val)

        model.module.load_state_dict(model_state_dict)
        model.module.eval()
        model._validate_anndata(adata)

        return model
