import numpy as np
import pytest

from scvi_neural_ode.data import synthetic_iid


@pytest.fixture
def adata():
    adata = synthetic_iid(batch_size=128, n_batches=2)  # length 256
    raw_counts = adata.X.copy()
    adata.layers["counts"] = raw_counts
    adata.obs["cont1"] = np.random.normal(size=adata.n_obs)
    adata.obs["cat1"] = np.random.randint(0, 5, size=adata.n_obs)

    return adata
