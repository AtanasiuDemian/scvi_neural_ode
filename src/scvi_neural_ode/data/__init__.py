from ._anndata import (
    add_categorical_obs,
    add_mito_frac,
    add_raw_proportions,
    get_from_registry,
    make_categorical_covariate,
    register_tensor_from_anndata,
    setup_data_registry,
    transfer_umap_to_anndata,
    update_obs_colors,
)
from ._anndataset import AnnDataSet
from ._data_modules import ConditionalAnnDataModule, LightningAnnDataModule
from ._datasets import synthetic_iid
from ._loaders import AnnDataLoader, ConcatAnnDataLoader

__all__ = [
    "add_categorical_obs",
    "add_raw_proportions",
    "get_from_registry",
    "add_mito_frac",
    "make_categorical_covariate",
    "register_tensor_from_anndata",
    "setup_data_registry",
    "transfer_umap_to_anndata",
    "update_obs_colors",
    "synthetic_iid",
    "AnnDataSet",
    "ConditionalAnnDataModule",
    "LightningAnnDataModule",
    "AnnDataLoader",
    "ConcatAnnDataLoader",
]
