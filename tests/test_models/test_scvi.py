import os

import numpy as np
import pytest
import torch

from scvi_neural_ode.data import setup_data_registry, synthetic_iid
from scvi_neural_ode.models import SCVI


def test_scvi():
    adata = synthetic_iid()
    n_latent = 5
    model = SCVI(adata, n_latent=n_latent)
    model.train(n_epochs=1, train_size=0.5, lr=1e-3)

    # tests __repr__
    print(model)

    assert model.is_trained is True
    n_latent = model.module.n_latent
    z = model.get_latent_representation()
    assert z.shape == (adata.n_obs, n_latent)
    model.get_elbo()
    model.get_reconstruction_error()
    model.get_reconstruction_error(return_mean=False)
    p = model.get_normalized_expression()
    assert p.shape == adata.shape
    model.posterior_predictive_sample()
    model.get_denoised_samples()
    param_dict = model.get_likelihood_parameters()
    assert param_dict["means"].shape == (adata.n_obs, adata.n_vars)
    assert param_dict["variances"].shape == (adata.n_obs, adata.n_vars)
    assert param_dict["dispersions"].shape == (adata.n_obs, adata.n_vars)
    model.posterior_z_params()

    adata2 = synthetic_iid()
    # test methods with a different dataset
    model.get_elbo(adata2)
    model.get_reconstruction_error(adata2)
    model.get_latent_representation(adata2)
    model.get_normalized_expression(adata2)
    model.posterior_predictive_sample(adata2)
    model.get_denoised_samples(adata2)
    model.get_likelihood_parameters(adata2)
    model.get_feature_correlation_matrix()

    z = model.get_latent_representation(adata2, indices=[2, 3, 4])
    assert z.shape == (3, n_latent)
    p = model.get_normalized_expression(adata2, indices=[3, 6, 7])
    assert p.shape == (3, adata2.n_vars)
    sample = model.posterior_predictive_sample(adata2, indices=[2, 3, 4])
    assert sample.shape == (3, adata2.n_vars)
    sample = model.get_denoised_samples(adata2, indices=[2, 3, 4])
    assert sample.shape == (3, adata2.n_vars)
    model.get_feature_correlation_matrix(adata2, indices=[1, 2, 3])
    model.posterior_z_params(adata2, indices=[0, 1, 2, 3])

    gene_list = ["1", "2"]
    p = model.get_normalized_expression(adata2, indices=[3, 4, 5], gene_list=gene_list)
    assert p.shape == (3, 2)
    model.get_normalized_expression(
        adata2,
        indices=[3, 4, 5],
        gene_list=gene_list,
        size_factor=50,
        return_numpy=False,
    )
    sample = model.posterior_predictive_sample(adata2, indices=[1, 2, 3], gene_list=gene_list)
    assert sample.shape == (3, 2)
    model.get_denoised_samples(adata2, indices=[1, 2, 3], size_factor=1)

    # test VAMP prior
    model.get_vamp_prior(p=10)

    # test training with full dataset
    model = SCVI(adata2, n_latent=n_latent)
    model.train(1, train_size=1)

    # WHY WOULD THIS trigger an assertion error?
    with pytest.raises(AssertionError):
        model.train(n_epochs=1, data_module_kwargs={"shuffle": False})

    model_outputs = model.forward_pass()
    assert model_outputs["x_rate"].shape == (model.adata.n_obs, model.adata.n_vars)
    assert model_outputs["x_log_disp"].shape == (model.adata.n_vars,)
    assert model_outputs["z_sample"].shape == (model.adata.n_obs, n_latent)
    z1 = model.get_latent_representation(use_mean=True)
    z2 = model_outputs["z_mean"]
    # np.testing.assert_allclose(z1, z2, rtol=1e-5)  # FIX THIS

    model_outputs = model.forward_pass(indices=[1, 2, 3, 4])
    assert model_outputs["x_rate"].shape == (4, model.adata.n_vars)
    assert model_outputs["z_sample"].shape == (4, n_latent)


def test_saving_loading(save_path):
    adata = synthetic_iid()
    n_latent = 5
    model = SCVI(adata, n_latent=n_latent)
    model.train(n_epochs=1, train_size=0.5, lr=1e-3)
    dir_path = os.path.join(save_path, "scvi")
    model.save(dir_path=dir_path, save_anndata=False)

    loaded_scvi = SCVI.load(dir_path=dir_path, adata=adata)
    loaded_scvi.get_latent_representation()
    loaded_scvi.get_normalized_expression()
    loaded_scvi.get_denoised_samples()


def test_differential_expression():
    adata = synthetic_iid(n_batches=2)
    adata.obs["cat"] = np.random.choice(["a", "b", "c"], size=adata.n_obs)
    model = SCVI(adata)
    model.train(1)
    N_samples = 50
    eps = 1e-4
    delta = 1
    kwargs = {"eps": eps, "delta": delta, "N_samples": N_samples}
    de_df, lfc_df = model.differential_expression(
        groupby="cat", group1=["a", "b"], group2="c", **kwargs
    )
    assert len(de_df) == adata.n_vars
    assert lfc_df.shape == (N_samples, adata.n_vars)

    indices1 = adata.obs.cat == "a"
    indices2 = adata.obs.cat == "b"
    model.differential_expression(indices1=indices1, indices2=indices2, **kwargs)
    with pytest.raises(ValueError):
        # Need to input indices OR group key.
        model.differential_expression(indices1=indices1, indices2=indices2, groupby="cat", **kwargs)

    rates = model.get_normalized_expression()
    model.differential_expression(
        rates=rates, groupby="cat", group1=["a", "b"], group2="c", **kwargs
    )
    del kwargs["N_samples"]
    model.differential_expression(
        indices1=np.array([0, 1, 2]), indices2=np.array([3, 4, 5]), N_samples=None, **kwargs
    )
    with pytest.raises(ValueError):
        # When comparing groups, need to enter `N_samples`.
        model.differential_expression(
            groupby="cat", group1=["a", "b"], group2="c", N_samples=None, **kwargs
        )
        # Throw error if groups overlap.
        model.differential_expression(groupby="cat", group1=["a", "b"], group2=["c", "b"], **kwargs)
        # Overlapping indices
        model.differential_expression(indices1=[0, 1, 2], indices2=[0, 1, 2], **kwargs)

    kwargs["N_samples"] = 250
    # Test summary function
    model.differential_expression(
        indices1=indices1, indices2=indices2, lfc_summary_fn=lambda x: np.abs(np.mean(x)), **kwargs
    )
    # Test decoding with covariates
    decoder_covariates = {"batch": 0}
    model.differential_expression(
        groupby="cat", group1="a", group2="b", decoder_covariates=decoder_covariates, **kwargs
    )


def test_conditional_scvi():
    adata = synthetic_iid(n_batches=2, n_labels=0)
    adata.obs["batch"] = np.random.choice(range(4), size=adata.n_obs)
    model = SCVI(adata, dispersion="batch-gene")
    model.train(1)
    model.get_latent_representation()
    model.get_normalized_expression()
    model.posterior_predictive_sample()
    model.get_denoised_samples()
    model.get_feature_correlation_matrix()
    model.get_likelihood_parameters()

    z = torch.from_numpy(model.get_latent_representation())
    model.decode_with_covariates(z, batch=1)

    # Decode with specific covariates directly via `forward_pass`.
    model.forward_pass(batch=1)
    # Throw error if covariate value is inappropriate.
    with pytest.raises(RuntimeError):
        model.forward_pass(batch=8)
    # This will throw a logger warning.
    model.forward_pass(batch=0, r=5)


def test_extra_covariates():
    adata = synthetic_iid(n_batches=2, n_labels=0)
    adata.obs["cont1"] = np.random.normal(size=adata.n_obs)
    adata.obs["cat1"] = np.random.randint(0, 5, size=adata.n_obs)
    setup_data_registry(
        adata, batch_key="batch", extra_cat_covs=["cat1"], extra_cont_covs=["cont1"]
    )
    model = SCVI(adata)
    model.train(1)

    z = model.get_latent_representation()
    p = model.decode_with_covariates(z, batch=0, cat1=3, cont1=adata.obs_vector("cont1"))
    assert p.shape == adata.shape

    adata.obs["cont2"] = np.random.normal(size=adata.n_obs)
    adata.obs["cat2"] = np.random.randint(0, 5, size=adata.n_obs)
    setup_data_registry(
        adata,
        batch_key="batch",
        extra_cat_covs=["cat1", "cat2"],
        extra_cont_covs=["cont1", "cont2"],
    )
    model = SCVI(adata)
    model.train(1)

    model.get_latent_representation()
    model.get_normalized_expression()
    model.posterior_predictive_sample()
    model.get_denoised_samples()
    model.get_elbo()
    model.get_reconstruction_error()

    # We have covariates batch, cat1, cat2, cont1, cont2
    # Decode with constant values for batch, cat1, cont1; default values for cat2; custom values for cont2
    model.decode_with_covariates(
        z,
        batch=0,
        cat1=2,
        cont1=0.1234,
        cat2=adata.obs_vector("cat2"),
        cont2=np.random.normal(size=adata.n_obs),
    )
    with pytest.raises(ValueError):
        model.decode_with_covariates(z, batch=0)

    # Decode with specific covariates directly via `forward_pass`.
    # Will use the metadata for all other covariates.
    model.forward_pass()
    model.forward_pass(cat2=3, batch=0)
    model.forward_pass(cat2=3, batch=0, cont1=np.random.normal(size=adata.n_obs))
    model.forward_pass(indices=[2, 3, 4], cat2=3, batch=0)
    # Throw error if custom array input does not match number of cells.
    with pytest.raises(RuntimeError):
        model.forward_pass(indices=[2, 3, 4], cont1=np.random.normal(size=4))
