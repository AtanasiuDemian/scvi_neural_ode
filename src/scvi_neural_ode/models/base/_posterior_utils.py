from typing import List, Optional

import numpy as np
import torch
from torch.distributions import Gamma, Poisson

from scvi_neural_ode import _CONSTANTS
from scvi_neural_ode.data import AnnDataLoader
from scvi_neural_ode.distributions import NB

# NOTE: These functions all assume "x_rate" denotes normalized gene proportions on the probability simplex.


def _get_normalized_expression(
    dataloader: AnnDataLoader,
    module,
    size_factor: float = 1.0,
    gene_list: Optional[List] = None,
    **module_kwargs,
):
    if gene_list is None:
        gene_mask = slice(None)
    else:
        all_genes = dataloader.dataset.adata.var_names
        gene_mask = [True if gene in gene_list else False for gene in all_genes]

    exprs = []
    for tensors in dataloader:
        module_outputs = module(tensors, **module_kwargs)
        rate = module_outputs["x_rate"][..., gene_mask].cpu().numpy()
        rate *= size_factor
        exprs += [rate]
    exprs = np.concatenate(exprs, axis=0)

    return exprs


def _posterior_predictive_sample(
    dataloader: AnnDataLoader,
    module,
    gene_list: Optional[List] = None,
    **module_kwargs,
):
    if gene_list is None:
        gene_mask = slice(None)
    else:
        all_genes = dataloader.dataset.adata.var_names
        gene_mask = [True if gene in gene_list else False for gene in all_genes]

    exprs = []
    for tensors in dataloader:
        x = tensors[_CONSTANTS.X_KEY]
        library = torch.sum(x, dim=1, keepdim=True)

        module_outputs = module(tensors, **module_kwargs)
        x_rate = module_outputs["x_rate"]
        x_log_disp = module_outputs["x_log_disp"]

        if not hasattr(module, "gene_distribution") or module.gene_distribution == "NB":
            # x_log_disp is always in R range.
            # NOTE: There aren't only single cell models that use this function, so some may not have nb param tuple.
            # By default assume a (mean, dispersion) parameterization.
            if hasattr(module, "nb_p") and module.nb_p:
                dist = NB(mu=library * x_rate, logits=x_log_disp)
            else:
                dist = NB(mu=library * x_rate, theta=torch.exp(x_log_disp))
        elif module.gene_distribution == "Poisson":
            dist = Poisson(library * x_rate + 1e-8)
        else:
            raise ValueError("Gene distribution not recognized.")

        expr_ = dist.sample()[:, gene_mask].cpu().numpy()
        exprs += [expr_]
    exprs = np.concatenate(exprs, axis=0)

    return exprs


def _denoised_samples(
    dataloader: AnnDataLoader,
    module,
    size_factor: float = 1.0,
    **module_kwargs,
):
    exprs = []
    for tensors in dataloader:
        x = tensors[_CONSTANTS.X_KEY]
        module_outputs = module(tensors, **module_kwargs)
        x_rate = module_outputs["x_rate"]
        x_rate *= size_factor
        x_log_disp = module_outputs["x_log_disp"]
        if x_log_disp.ndim != 2:
            x_log_disp = torch.ones_like(x).to(x_log_disp.device) * x_log_disp

        if module.theta:
            x_disp = torch.exp(x_log_disp)  # exp(log(theta))
            shape = x_disp
            gamma_rate = x_disp / x_rate
        elif module.nb_p:
            gamma_rate = torch.exp(x_log_disp)  # exp(log(p/1-p))
            shape = x_rate * gamma_rate
        samples = Gamma(concentration=shape, rate=gamma_rate).sample().cpu().numpy()
        exprs += [samples]

    return np.concatenate(exprs, axis=0)


def _get_likelihood_params(dataloader: AnnDataLoader, module, **module_kwargs):
    """
    Returns mean, variance and dispersion of the Negative Binomial likelihood.
    """
    # Parameters are interpreted differently depending on the NB parameterization.
    use_param_1 = module.mu and module.theta
    use_param_2 = module.mu and module.nb_p

    if use_param_1:

        def _mean_func(mu, _):
            return mu

        def _disp_func(_, log_theta):
            return torch.exp(log_theta)

        def _var_func(mu, log_theta):
            return mu**2 / torch.exp(log_theta) + mu  # mu**2/theta + mu

    elif use_param_2:

        def _mean_func(mu, _):
            return mu

        def _disp_func(mu, logits):
            return mu * torch.exp(logits)  # mu * p / (1-p)

        def _var_func(mu, logits):
            return mu / torch.nn.functional.sigmoid(logits)  # mu/p

    else:  # r, p

        def _mean_func(r, logits):
            return r * torch.exp(-logits)  # r * (1-p)/p

        def _disp_func(r, _):
            return r

        def _var_func(r, logits):
            return _mean_func(r, logits) / torch.nn.functional.sigmoid(logits)  # r*(1-p)/p**2

    # x_log_disp can be 1- or 2- dimensional depending on whether we use sample information (batch-gene etc.)
    mean_list, var_list, disp_list = [], [], []
    for tensors in dataloader:
        m = tensors["X"].shape[0]
        outputs = module(tensors, **module_kwargs)
        x_rate = outputs["x_rate"]
        x_log_disp = outputs["x_log_disp"]

        disp_term = _disp_func(x_rate, x_log_disp)
        if disp_term.ndim == 1:
            disp_term = disp_term.repeat(m, 1)
        disp_list += [disp_term.cpu().numpy()]
        mean_list += [_mean_func(x_rate, x_log_disp).cpu().numpy()]
        var_list += [_var_func(x_rate, x_log_disp).cpu().numpy()]

    output_dict = {}
    output_dict["means"] = np.concatenate(mean_list, axis=0)
    output_dict["variances"] = np.concatenate(var_list, axis=0)
    output_dict["dispersions"] = np.concatenate(disp_list, axis=0)

    return output_dict
