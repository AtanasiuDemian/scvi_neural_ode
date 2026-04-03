import numpy as np

from scvi_neural_ode import _CONSTANTS
from scvi_neural_ode.data import synthetic_iid
from scvi_neural_ode.models import ODESCVI, BranchingCondODESCVI, CondODESCVI


def test_trajectory_for_each_label():
    batch_size, n_batches = 128, 4
    n_latent = 10
    adata = synthetic_iid(n_batches=n_batches, batch_size=batch_size)
    model = ODESCVI(adata, n_latent=n_latent)
    model.train(1)
    # Test get_time
    T = model.get_time()
    assert T.size == adata.n_obs
    # Test get_trajectory_output
    output = model.get_trajectory_output()
    label_output = output[0]
    assert "indices" in label_output.keys()
    indices = label_output["indices"]
    assert len(label_output["T"]) == len(indices)
    assert label_output["pred_z"].shape == (len(indices), n_latent)
    assert label_output["pred_x"].shape == (len(indices), adata.n_vars)
    # Test get_latent_representation
    latents = model.get_latent_representation()
    assert latents.shape == (adata.n_obs, n_latent)
    adata_indices = [2, 4, 6]
    latents = model.get_latent_representation(indices=adata_indices)
    assert latents.shape == (len(adata_indices), n_latent)
    # Test get_normalized_expression
    expr = model.get_normalized_expression()
    assert expr.shape == (adata.n_obs, adata.n_vars)
    expr = model.get_normalized_expression(indices=adata_indices)
    assert expr.shape == (len(adata_indices), adata.n_vars)


def test_conditional_model():
    n_batches = 4
    n_latent = 10
    adata = synthetic_iid(n_batches=n_batches)
    model = CondODESCVI(adata, n_latent=n_latent)
    model.train(1)
    model.get_time()
    model = CondODESCVI(adata, n_latent=n_latent, kl_scaling=5)
    model.train(1)

    # Extra covariates
    adata.obs["cont1"] = np.random.normal(size=adata.n_obs)
    adata.obs["cat1"] = np.random.randint(0, 5, size=adata.n_obs)
    adata.uns["data_registry"].update({_CONSTANTS.CONT_COV_KEY: ["obs", "cont1"]})
    adata.uns["data_registry"].update({_CONSTANTS.CAT_COV_KEY: ["obs", "cat1"]})
    model = CondODESCVI(adata, n_latent=n_latent, kl_scaling=5)
    model.train(1)
    model.get_time()
    model.forward_pass()

    model = CondODESCVI(adata, n_latent=n_latent, dispersion="batch-gene")
    model.train(1)
    model.get_time()
    model.forward_pass()


def test_branching_model():
    n_batches = 4
    adata = synthetic_iid(n_batches=n_batches)
    n_traj = 4
    CAT_KEY = "branch_ix"
    gt_branches = np.random.choice(range(n_traj), size=adata.n_obs)
    adata.obs[CAT_KEY] = gt_branches
    model = BranchingCondODESCVI(
        adata=adata,
        n_latent=5,
        n_cats=n_traj,
        CAT_KEY=CAT_KEY,
        dispersion="batch-gene",
    )
    model.train(1)
    model.forward_pass()
    branches = model.get_branch_categories()
    np.testing.assert_array_equal(gt_branches, branches)
