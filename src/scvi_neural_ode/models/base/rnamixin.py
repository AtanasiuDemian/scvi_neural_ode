from typing import Literal, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from anndata import AnnData
from scipy.stats import spearmanr

from scvi_neural_ode.models.base._posterior_utils import (
    _denoised_samples,
    _get_likelihood_params,
    _get_normalized_expression,
    _posterior_predictive_sample,
)


class RNASeqMixin:
    @torch.no_grad()
    def get_normalized_expression(
        self,
        adata: Optional[AnnData] = None,
        indices: Optional[Sequence[int]] = None,
        gene_list: Optional[Sequence[str]] = None,
        size_factor: float = 1.0,
        batch_size: Optional[int] = None,
        return_numpy: bool = True,
        **module_kwargs,
    ):
        if adata is None:
            adata = self.adata
        scdl = self._make_data_loader(
            adata=adata, indices=indices, batch_size=batch_size, shuffle=False
        )
        if indices is None:
            indices = np.arange(len(adata))

        exprs = _get_normalized_expression(
            dataloader=scdl,
            module=self.module,
            size_factor=size_factor,
            gene_list=gene_list,
            **module_kwargs,
        )

        if not return_numpy:
            return pd.DataFrame(
                exprs,
                columns=adata.var_names if gene_list is None else gene_list,
                index=adata.obs_names[indices],
            )
        else:
            return exprs

    @torch.no_grad()
    def posterior_predictive_sample(
        self,
        adata: Optional[AnnData] = None,
        indices: Optional[Sequence[int]] = None,
        gene_list: Optional[Sequence[str]] = None,
        batch_size: Optional[int] = None,
        return_numpy: bool = True,
        **module_kwargs,
    ):
        """
        Generate observation samples from the posterior predictive distribution.
        """
        if adata is None:
            adata = self.adata
        scdl = self._make_data_loader(
            adata=adata, indices=indices, batch_size=batch_size, shuffle=False
        )
        if indices is None:
            indices = np.arange(len(adata))

        exprs = _posterior_predictive_sample(
            dataloader=scdl,
            module=self.module,
            gene_list=gene_list,
            **module_kwargs,
        )

        if not return_numpy:
            return pd.DataFrame(
                exprs,
                columns=adata.var_names if gene_list is None else gene_list,
                index=adata.obs_names[indices],
            )
        else:
            return exprs

    @torch.no_grad()
    def get_denoised_samples(
        self,
        adata: Optional[AnnData] = None,
        indices: Optional[Sequence[int]] = None,
        batch_size: int = 64,
        size_factor: float = 1000,
        **module_kwargs,
    ):
        # When adding n_samples arg, samples should come from the encoder.
        scdl = self._make_data_loader(adata=adata, indices=indices, batch_size=batch_size)

        return _denoised_samples(
            dataloader=scdl,
            module=self.module,
            size_factor=size_factor,
            **module_kwargs,
        )

    @torch.no_grad()
    def get_feature_correlation_matrix(
        self,
        adata: Optional[AnnData] = None,
        indices: Optional[Sequence[int]] = None,
        batch_size=64,
        size_factor=1000,
        correlation_type: Literal["spearman", "pearson"] = "pearson",
        center_values: bool = False,
    ):
        if adata is None:
            adata = self.adata

        denoised_samples = self.get_denoised_samples(
            adata=adata, indices=indices, batch_size=batch_size, size_factor=size_factor
        )
        if center_values is True:
            denoised_samples = (denoised_samples - denoised_samples.mean(0)) / denoised_samples.std(
                0
            )

        if correlation_type == "pearson":
            corr_matrix = np.corrcoef(denoised_samples, rowvar=False)
        elif correlation_type == "spearman":
            corr_matrix = spearmanr(denoised_samples)
        else:
            raise ValueError("Unknown correlation type, choose either 'spearman' or 'pearson'")

        return pd.DataFrame(corr_matrix, index=adata.var_names, columns=adata.var_names)

    @torch.no_grad()
    def get_likelihood_parameters(
        self,
        adata: Optional[AnnData] = None,
        indices: Optional[Sequence[int]] = None,
        batch_size: Optional[int] = None,
        return_numpy: bool = True,
        **module_kwargs,
    ):
        if adata is None:
            adata = self.adata
        if indices is None:
            indices = np.arange(len(adata))
        scdl = self._make_data_loader(
            adata=adata, indices=indices, batch_size=batch_size, shuffle=False
        )

        exprs = _get_likelihood_params(dataloader=scdl, module=self.module, **module_kwargs)

        if not return_numpy:
            return pd.DataFrame(
                exprs,
                columns=adata.var_names,
                index=adata.obs_names[indices],
            )
        else:
            return exprs
