import numpy as np
from anndata import AnnData

from ._anndata import setup_data_registry


def synthetic_iid(
    batch_size: int = 128,
    n_genes: int = 100,
    n_batches: int = 2,
    n_labels: int = 0,
):
    data = np.random.negative_binomial(5, 0.3, size=(batch_size * n_batches, n_genes))
    batch = []
    extra_cat_covs = None
    for i in range(n_batches):
        # batch += ["batch_{}".format(i)] * batch_size
        batch += [i] * batch_size
    adata = AnnData(data)
    adata.obs["batch"] = batch
    if n_labels > 0:
        labels = np.random.randint(0, n_labels, size=(batch_size * n_batches,))
        adata.obs["label"] = labels
        extra_cat_covs = ["label"]
    adata.obs["n_counts"] = adata.X.sum(1)
    setup_data_registry(adata, batch_key="batch", extra_cat_covs=extra_cat_covs)

    return adata
