import logging
from typing import Callable, List, Optional, Union

import numpy as np
import numpy.typing as npt
import pandas as pd

logger = logging.getLogger(__name__)


def LFC(x, y, eps):
    return np.log2(x + eps) - np.log2(y + eps)


class DifferentialMixin:
    def differential_expression(
        self,
        eps: float,
        delta: float,
        rates: Optional[npt.NDArray[np.float32]] = None,
        groupby: Optional[str] = None,
        group1: Optional[Union[str, List[str]]] = None,
        group2: Optional[Union[str, List[str]]] = None,
        indices1: Optional[npt.NDArray[Union[np.float32, np.int32, np.bool]]] = None,
        indices2: Optional[npt.NDArray[Union[np.float32, np.int32, np.bool]]] = None,
        N_samples: Optional[int] = None,
        rates_transform_fn: Optional[Callable] = None,
        lfc_summary_fn: Optional[Callable] = None,
        fdr: float = 0.05,
        decoder_covariates: Optional[dict] = None,
    ):
        """
        Implement the Differential Expression method adapted from Boyeau et al, 2022.

        Input
        -----
        eps
            Offset used in computing logfold change to remove genes with low proportions.
        delta
            Positive threshold on |LFC|.
        rates
            Array of gene proportions.
        groupby
            Group identifier for cells.
        group1, group2
            If `groupby` is not None, compare cells from these 2 groups.
        indices1, indices2
            Indices in self.adata of 2 non-overlapping cell groups.
            If they have different lengths, then must select `N_samples`.
        N_samples
            Number of cells to choose uniformly at random.
        rates_transform_fn
            Function to apply to gene proportions.
        lfc_summary_fn
            Summary function for logfold change. Default is median.
        decoder_covariates
            Specific input covariates for the conditional decoder.

        Output
        ------
        de_df: pandas.DataFrame
            Dataframe with gene name rows, with columns:
                summary_LFC : value of summary LFC.
                probs : proportion of samples for which |LFC| >= delta.

        lfc_df: pandas.DataFrame
            DataFrame with all LFC values, columns are the gene names.
        """
        if groupby is not None and (indices1 is not None or indices2 is not None):
            raise ValueError("Pick either 'groupby' or custom array of indices.")
        if groupby is not None and (group1 is None or group2 is None):
            raise ValueError("Pick both 'group1' and 'group2'.")
        if groupby is None and (indices1 is None or indices2 is None):
            raise ValueError("If `groupby` is None, then pick both 'indices1' and 'indices2'.")

        if groupby is not None:
            if isinstance(group1, str):
                group1 = [group1]
            if isinstance(group2, str):
                group2 = [group2]

            if np.intersect1d(group1, group2).size:
                raise ValueError("Group labels must be non-overlapping.")
            if N_samples is None:
                raise ValueError("If using `groupby` then also need to provide `N_samples`.")
            indices1 = np.where(self.adata.obs[groupby].isin(group1))[0]
            indices2 = np.where(self.adata.obs[groupby].isin(group2))[0]
        else:
            if indices1.dtype == bool:
                indices1 = np.where(indices1)[0]
            if indices2.dtype == bool:
                indices2 = np.where(indices2)[0]
            if np.intersect1d(indices1, indices2).size:
                raise ValueError("If using indices, then they must be non-overlapping.")

        if not indices1.size or not indices2.size:
            raise ValueError("At least one of the group arrays is empty.")

        if N_samples is not None:
            ix1 = np.random.choice(indices1, replace=True, size=N_samples)
            ix2 = np.random.choice(indices2, replace=True, size=N_samples)
        else:
            assert len(indices1) == len(
                indices2
            ), "If `N_samples` argument is not provided, then `indices1` and `indices2` must have the same length."
            ix1, ix2 = indices1, indices2

        if rates is not None:
            rates1, rates2 = rates[ix1], rates[ix2]
        else:
            decoder_covariates = {} if decoder_covariates is None else decoder_covariates
            rates1 = self.forward_pass(indices=ix1, **decoder_covariates)["x_rate"]
            rates2 = self.forward_pass(indices=ix2, **decoder_covariates)["x_rate"]

        if rates_transform_fn is not None:
            rates1 = rates_transform_fn(rates1)
            rates2 = rates_transform_fn(rates2)

        self.delta = delta
        self.eps = eps
        lfc = LFC(rates1, rates2, eps)  # N_samples, n_genes
        lfc_df = pd.DataFrame(lfc, columns=self.adata.var_names)
        if lfc_summary_fn is None:
            lfc_summary = np.median(lfc, axis=0)
        else:
            lfc_summary = np.apply_along_axis(lfc_summary_fn, axis=0, arr=lfc)

        de_df = pd.DataFrame(index=self.adata.var_names)
        de_df["summary_LFC"] = lfc_summary
        probs = np.maximum(np.mean(lfc >= delta, axis=0), np.mean(lfc <= -delta, axis=0))
        de_df["probs"] = probs
        de_df = de_df.sort_values(by="probs", ascending=False)
        de_df[f"is_de_fdr_{fdr}"] = _fdr_de_prediction(de_df["probs"], fdr=fdr)

        return de_df, lfc_df

    def plot_volcano(self, de_df: pd.DataFrame, annotate_de_genes: bool = False, **kwargs):
        if "de_key" not in kwargs.keys():
            # By default plot the FDR key
            # if we have more than 1 FDR key, then de_key in plot_volcano will be None.
            fdr_keys = [v for v in de_df.columns if v.startswith("is_de_fdr")]
            if len(fdr_keys) > 1:
                logger.warning(
                    f"There is more than 1 FDR key: {fdr_keys}. By default will plot volcano with de_key=None"
                    "unless you add a specific FDR key."
                )
                kwargs["de_key"] = None
            elif len(fdr_keys) == 1:
                key = fdr_keys[0]
                logger.info(f"Plotting volcano with de_key {key}.")
                kwargs["de_key"] = key
            else:
                logger.info("No FDR keys found. Plotting with default de_key=None.")
                kwargs["de_key"] = None

        # return plot_volcano(
        #     de_df=de_df,
        #     annotate_de_genes=annotate_de_genes,
        #     delta=self.delta,
        #     eps=self.eps,
        #     **kwargs,
        # )


def _fdr_de_prediction(posterior_probas: pd.Series, fdr: float = 0.05) -> pd.Series:
    """
    Taken from Boyeau et al, 2022
    https://github.com/scverse/scvi-tools/blob/main/src/scvi/model/base/_de_core.py#L114
    Commit 337ec87

    Compute posterior expected FDR and tag features as DE.
    """
    if not posterior_probas.ndim == 1:
        raise ValueError("posterior_probas should be 1-dimensional")
    original_index = posterior_probas.index
    sorted_pgs = posterior_probas.sort_values(ascending=False)
    cumulative_fdr = (1.0 - sorted_pgs).cumsum() / (1.0 + np.arange(len(sorted_pgs)))
    d = (cumulative_fdr <= fdr).sum()
    is_pred_de = pd.Series(np.zeros_like(cumulative_fdr).astype(bool), index=sorted_pgs.index)
    is_pred_de.iloc[:d] = True
    is_pred_de = is_pred_de.loc[original_index]
    return is_pred_de
