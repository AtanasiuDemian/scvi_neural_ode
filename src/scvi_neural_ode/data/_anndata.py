from typing import List, Literal, Optional

import numpy as np
import pandas as pd
from anndata import AnnData

from scvi_neural_ode import _CONSTANTS


def get_from_registry(adata: AnnData, key: str):
    """
    .uns['data_registry'] returns something like
    {'X': ['X', None], 'n_counts': ['obs', 'n_counts']}
    """
    attr_name, attr_key = adata.uns["data_registry"][key]
    data = getattr(adata, attr_name)

    if attr_key != "None":
        # NOTE: in scvi they write the string "None", because the model will always save "None" as a string.
        if isinstance(data, pd.DataFrame):
            data = data.loc[:, attr_key]
        else:
            data = data[attr_key]
    if isinstance(data, pd.Series):
        data = data.to_numpy().reshape(-1, 1)

    return data


def register_tensor_from_anndata(
    adata: AnnData,
    registry_key: str,
    adata_attr_name: Literal["obs", "var", "obsm", "varm", "uns"],
    adata_key_name: str,
    is_categorical: Optional[bool] = False,
):
    """
    Add tensor to AnnData data registry.
    """
    if is_categorical is True:
        if adata_attr_name != "obs":
            raise ValueError("Categorical handling implemented only for data in .obs")

    # Add step for making categorical columns.

    new_dict = {registry_key: [adata_attr_name, adata_key_name]}
    # Can also change the format of data registry to be like:
    # {registry_key: {'attr_name' : adata_attr_name, 'attr_key' : adata_key_name}}
    data_registry = adata.uns["data_registry"]
    data_registry.update(new_dict)

    # Add step to verify and correct data format.


def add_categorical_obs(
    adata: AnnData,
    vals: pd.DataFrame,
    cat_key_name: str,
):
    """
    Dataframe vals must share indices with adata.obs
    """
    isin_mask = vals.index.isin(adata.obs.index)

    # Assume not all observation indices of adata are in vals.
    adata.obs[cat_key_name] = np.nan
    adata.obs.loc[vals.index[isin_mask], cat_key_name] = vals.iloc[isin_mask, 0].values

    # NOTE: THIS MAPPING CAN BE PROBLEMATIC if you later change or add categories.
    adata.uns["{}_mapping".format(cat_key_name)] = (
        adata.obs[cat_key_name].astype("category").cat.categories.values
    )


def add_raw_proportions(adata: AnnData, layer_key: str = "raw"):
    """Normalize gene expression to proportions summing up to 1."""
    adata.layers[layer_key] = adata.X / adata.X.sum(1)[:, None]


def setup_data_registry(
    adata: AnnData,
    batch_key: Optional[str] = None,
    extra_cat_covs: Optional[List[str]] = None,
    extra_cont_covs: Optional[List[str]] = None,
):
    def _validate_keys(keys):
        for key in keys:
            if key not in adata.obs.columns:
                raise ValueError(f"Key {key} not found in .obs")

    data_registry = {_CONSTANTS.X_KEY: ["X", "None"]}
    if batch_key is not None:
        _validate_keys([batch_key])
        # Here assume batch key is only one covariate.
        data_registry.update({_CONSTANTS.BATCH_KEY: ["obs", batch_key]})
    if extra_cat_covs is not None:
        _validate_keys(extra_cat_covs)
        if len(extra_cat_covs) == 1:
            extra_cat_covs = extra_cat_covs[0]
        data_registry.update({_CONSTANTS.CAT_COV_KEY: ["obs", extra_cat_covs]})
    if extra_cont_covs is not None:
        _validate_keys(extra_cont_covs)
        if len(extra_cont_covs) == 1:
            extra_cont_covs = extra_cont_covs[0]
        data_registry.update({_CONSTANTS.CONT_COV_KEY: ["obs", extra_cont_covs]})

    adata.uns["data_registry"] = data_registry


def make_categorical_covariate(adata: AnnData, obs_name: str, attr_name: str):
    """
    Map a discrete-valued column into a categorical covariate of integer codes.

    Does NOT update data_registry.
    """
    adata.obs[attr_name] = pd.Categorical(adata.obs[obs_name]).codes


def transfer_umap_to_anndata(adata_source: AnnData, adata_target: AnnData):
    """
    Transfer UMAP coordinates to another anndata object.

    This comes in handy if we want to plot expression in UMAP of genes not included in the anndata the UMAP was computed on.
    """
    umap_df = pd.DataFrame(adata_source.obsm["X_umap"], index=adata_source.obs.index)
    adata_target.obsm["X_umap"] = umap_df.loc[adata_target.obs.index].values


def update_obs_colors(adata: AnnData, obs_name: str, new_colors_dict: dict):
    """
    Edit the colors corresponding to a column in .obs. Modifies anndata in-place.
    """
    mapping = pd.Categorical(adata.obs[obs_name]).categories.values
    uns_key = f"{obs_name}_colors"
    if uns_key in adata.uns.keys():
        old_colors_dict = dict(zip(mapping, adata.uns[uns_key]))
        adata.uns[uns_key] = [
            new_colors_dict[key] if key in new_colors_dict else old_colors_dict[key]
            for key in mapping
        ]
    else:
        print(f"Creating new key in .uns: {uns_key}")
        adata.uns[uns_key] = [new_colors_dict[key] for key in mapping]


def add_mito_frac(adata: AnnData, key="mito_frac"):
    """Compute proportion of mitochondrial expression. Modifies adata in-place."""
    mt_gene_mask = [gene.startswith(("Mt", "mt")) for gene in adata.var_names]
    adata.obs[key] = adata[:, mt_gene_mask].X.sum(1) / adata.X.sum(1)
