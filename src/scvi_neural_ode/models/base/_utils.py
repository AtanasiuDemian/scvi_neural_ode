import logging
import os
import pickle

import numpy as np
import torch
from anndata import read_h5ad

logger = logging.getLogger(__name__)


def _load_saved_files(
    dir_path: str,
    load_data: bool,
):
    """
    "Helper to load saved files."
    """
    setup_dict_path = os.path.join(dir_path, "attr.pkl")
    anndata_path = os.path.join(dir_path, "adata.h5ad")
    varnames_path = os.path.join(dir_path, "var_names.csv")
    model_path = os.path.join(dir_path, "model_params.pt")

    if os.path.exists(anndata_path) and load_data:
        anndata = read_h5ad(anndata_path)
    elif not os.path.exists(anndata_path) and load_data:
        raise ValueError("Saved path contains no anndata")
    else:
        anndata = None
    var_names = np.genfromtxt(varnames_path, delimiter=",", dtype="str")
    with open(setup_dict_path, "rb") as handle:
        attr_dict = pickle.load(handle)
    model_state_dict = torch.load(model_path)

    return attr_dict, var_names, model_state_dict, anndata


def _initialize_model(cls, anndata, attr_dict):
    if "init_params_" not in attr_dict.keys():
        raise ValueError("No init_params were saved by the model.")
    init_params = attr_dict.pop("init_params_")
    non_kwargs = {k: v for k, v in init_params.items() if not isinstance(v, dict)}
    kwargs = {k: v for k, v in init_params.items() if isinstance(v, dict)}
    kwargs = {k: v for (i, j) in kwargs.items() for (k, v) in j.items()}  # expanding, v is a dict.
    model = cls(anndata, **non_kwargs, **kwargs)

    return model


def _validate_var_names(anndata, source_var_names):
    user_var_names = anndata.var_names.astype(str)
    if not np.array_equal(source_var_names, user_var_names):
        logger.warning(
            "var_names of passed anndata does not match the var_names of "
            "the anndata used in training the model."
        )
