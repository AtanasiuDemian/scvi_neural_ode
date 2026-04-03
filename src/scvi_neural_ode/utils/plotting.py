from typing import List, Optional, Union, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
import seaborn as sns
from matplotlib.axes import Axes



def get_color_values(
    adata: AnnData, key: str, vals: Optional[List] = None, na_color: str = "lavender"
):
    """
    Given a 'key', assumed to be a column in adata.obs, return an array of
    corresponding colors for each value in vals. If vals is None, then return array
    of colors for all values in the column.
    """
    # First check whether `key` is Categorical, as we might want to have a specific category ordering. Only call pd.Categorical when it is not.
    if adata.obs[key].dtype == "category":
        key_categories = adata.obs[key].cat.categories
    else:
        key_categories = pd.Categorical(adata.obs[key]).categories
    color_key = "{}_colors".format(key)
    if color_key in adata.uns.keys():
        color_list = np.array(adata.uns["{}_colors".format(key)])
    else:
        color_list = np.array(
            [
                matplotlib.colors.rgb2hex(c)
                for c in plt.cm.rainbow(np.linspace(0, 1, len(key_categories)))
            ]
        )

    _row_col = lambda s: (
        color_list[np.where(key_categories == s)[0][0]] if not pd.isnull(s) else na_color
    )
    if vals is None:
        vals = adata.obs[key].values

    return np.vectorize(_row_col)(vals)


def plot_category(adata, col_name, cat_name=None, na_color="lavender", **scanpy_kwargs):
    if cat_name is None:
        cats = adata.obs[col_name].astype("category").cat.categories
    else:
        cats = [cat_name]

    col_names = []
    for cat_name in cats:
        plot_col = "plot_{}".format(cat_name)
        adata.obs[plot_col] = np.nan
        adata.obs.loc[adata.obs[col_name] == cat_name, plot_col] = cat_name
        cat_i = np.where(pd.Categorical(adata.obs[col_name]).categories == cat_name)[0][0]
        adata.uns["{}_colors".format(plot_col)] = [adata.uns["{}_colors".format(col_name)][cat_i]]
        col_names.append(plot_col)

    if "title" not in scanpy_kwargs:
        title = ["{} {}".format(col_name, cat_name) for cat_name in cats]
        scanpy_kwargs.update(title=title)
    sc.pl.umap(adata, color=col_names, legend_loc=None, na_color=na_color, **scanpy_kwargs)
    for col in col_names:
        del adata.obs[col]
        del adata.uns[f"{col}_colors"]


def stripplot_from_anndata(
    adata: AnnData,
    key: str,
    groupby: str,
    layer: str = "raw",
    categories_order: Optional[List[Union[str, int]]] = None,
    hue: Optional[str] = None,
    figsize: Optional[Tuple[float, float]] = None,
    ax: Optional[Axes] = None,
    title: Optional[str] = None,
    subsample: Optional[int] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    **stripplot_kwargs,
):
    """
    Categorical scatterplot using jitter.

    Input
    -----
    adata
        AnnData object.
    key
        Input for numerical variable to be plotted (on the y-axis).
    groupby
        Categorical variable to be plotted on the x-axis.
    layer
        Layer key in anndata.
    categories_order
        Custom order of categories of the `groupby` variable.
    hue
        Variable to split the data in multiple levels, for each category of `groupby` variable.
    figsize
        Figure size.
    ax
        Instance of matplotlib.Axes.
    title
        Figure title.
    subsample
        Number of cells to subsample to, to reduce computation time. If None, plot all cells
    xlabel, ylabel
        Labels for x-axis and y-axis.
    **stripplot_kwargs
        Keyword arguments for seaborn.stripplot
    """
    expr = adata.obs_vector(key, layer=layer)
    df = pd.DataFrame(expr, columns=[key])
    df[groupby] = adata.obs_vector(groupby)
    if hue is not None:
        df[hue] = adata.obs_vector(hue)
    palette = (
        adata.uns.get(f"{groupby}_colors", None)
        if hue is None
        else adata.uns.get(f"{hue}_colors", None)
    )

    if subsample is not None and subsample <= len(df):
        subsample_mask = np.random.choice(len(df), size=subsample, replace=False)
        df = df.iloc[subsample_mask]

    if categories_order is None:
        categories_order = pd.Categorical(adata.obs[groupby]).categories.values

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    sns.stripplot(
        data=df,
        x=groupby,
        order=categories_order,
        y=key,
        hue=hue,
        dodge=True,
        palette=palette,
        ax=ax,
        **stripplot_kwargs,
    )
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


def plot_volcano(
    de_df: pd.DataFrame,
    lfc_key: str = "summary_LFC",
    probs_key: str = "probs",
    de_key: Optional[str] = None,
    annotate_de_genes: bool = True,
    plot_log_probs: bool = False,
    gene_colors: Optional[List[str]] = None,
    fontsize: float = 8.0,
    delta: Optional[float] = None,
    eps: Optional[float] = None,
    sig_gene_color: str = "red",
    nonsig_gene_color: str = "black",
    title: Optional[str] = None,
    size=15,
    figsize: Optional[Tuple[float, float]] = None,
    ax: Optional[Axes] = None,
    return_ax: bool = False,
):
    """
    Volcano plot highlighting significant genes.

    Input
    -----
    de_df
        DataFrame of gene information. Index will be the genes names, and it contains at least 2
        columns: LFC and probs values. Optional boolean column `de_key` will be used to classify
        gene as significant. If `de_key` is None, select genes that are above delta in absolute value.
    lfc_key
        Key for column of summary LFC values in de_df.
    probs_key
        Key for column of probs i.e. p(|LFC| >= delta) in de_df.
    annotate_de_genes
        Whether to annotate the significant genes.
    plot_log_probs
        Whether to plot -log10(1-probs).
    gene_colors
        Array of custom gene specific colors, this will be used to color the significant genes.
        Must be the same length as lfc_vals.
    fontsize
        Font size for text annotations, if annotate_de_genes=True.
    delta
        Delta threshold on |LFC|.
    eps
        Offset used in computing log fold change.
    sig_gene_color
        Color to use for significant genes. Overriden if gene_colors is not None.
    nonsig_gene_color
        Color to use for non-significant genes.
    title
        Title of figure.
    size
        Point size.
    figsize
        Custom figure size.
    ax
        Instance of matplotlib Axes object.
    return_ax
        Whether to return Axes object.
    """
    lfc_vals = de_df[lfc_key].values  # summary LFC values per gene
    probs = de_df[probs_key].values
    if de_key is None:
        assert (
            delta is not None
        ), "If `de_key` is None, then significance is assessed based on delta, hence you need to provide delta."
        de_mask = np.abs(lfc_vals) >= delta
    else:
        de_mask = de_df[de_key].values

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    assert len(lfc_vals) == len(probs) and len(probs) == len(
        de_mask
    ), f"Inputs lfc_vals ({len(lfc_vals)}), p_vals ({len(probs)}) and de_mask ({len(de_mask)}) must have the same size."
    if gene_colors is not None:
        assert len(gene_colors) == len(
            lfc_vals
        ), f"Input gene_colors ({len(gene_colors)}) must have the same size as lfc_vals, p_vals and de_mask ({len(lfc_vals)})."
        sig_gene_color = gene_colors[de_mask]

    yvals = probs if not plot_log_probs else -np.log10(1 - probs)
    ax.scatter(lfc_vals[de_mask], yvals[de_mask], c=sig_gene_color, s=size)
    ax.scatter(lfc_vals[~de_mask], yvals[~de_mask], c=nonsig_gene_color, s=size)
    if annotate_de_genes:
        # Simply get `gene_names` from de_df
        gene_names = de_df.index.values
        for i in range(sum(de_mask)):
            ax.annotate(
                gene_names[de_mask][i],
                (lfc_vals[de_mask][i], yvals[de_mask][i]),
                fontsize=fontsize,
            )

    ylabel = r"$p(|LFC| \geq \delta)$" if not plot_log_probs else r"$-log_{10}(pvals)$"
    ax.set_xlabel("LFC")
    ax.set_ylabel(ylabel)
    if title is None and delta is not None and eps is not None:
        title = r"$\delta={}, \varepsilon={}$".format(delta, eps)
    ax.set_title(title)
    if return_ax:
        return ax