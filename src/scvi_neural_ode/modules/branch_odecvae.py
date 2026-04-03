from typing import Optional

import torch
import torch.nn as nn
from torch.distributions import Normal, kl_divergence
from torchdiffeq import odeint

from scvi_neural_ode import _CONSTANTS
from scvi_neural_ode.distributions import NB
from scvi_neural_ode.modules import ODECVAE
from scvi_neural_ode.nn import ODEfunc
from scvi_neural_ode.utils.ode import get_step_size, unique_index


class BranchingODECVAE(ODECVAE):
    """
    Branching model using neural ODEs in latent space.

    Assumes data is split into multiple categories, each corresponding to one ODE trajectory.
    There is a root trajectory (0) and 1, 2, ..., n_cats - 1 lineages that branch off.

    This module assumes all trajectory memberships are known and assigned.

    All lineage trajectories have the same starting point, defined as the last cell (i.e. with highest pseudotime)
    in the root trajectory.

    Input
    ------
    n_input
        Number of genes.
    n_latent
        Number of latent variables.
    n_cats
        Number of categories: root + lineage trajectories.
    CAT_KEY
        Category column key in anndata .obs
    iroot
        Root cell in the root trajectory. If not None, each minibatch will contain this cell and it will be used as
        initial value for the ODE.
    """

    def __init__(
        self,
        n_input: int,
        n_latent: int,
        n_cats: int,
        CAT_KEY: str,
        iroot: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(n_input=n_input, n_latent=n_latent, **kwargs)
        self.n_cats = n_cats
        self.CAT_KEY = CAT_KEY
        self.iroot = iroot
        self.ODEfuncs = nn.ModuleList(
            [
                ODEfunc(
                    n_input=n_latent,
                    n_hidden=64,
                    n_output=n_latent,
                    activation_fn=nn.ELU,
                )
                for _ in range(self.n_cats)
            ]
        )
        # self.pz_scale = nn.Parameter(torch.randn(self.n_cats, self.n_latent))

    def _ODEfunc_batch_wrapper(self, t, x):
        # NOTE: only to be used by odeint. Then we always have x.ndim = 1

        return self.ODEfunc(t, x.unsqueeze(0), self.curr_batch).squeeze(0)

    def forward(self, tensors: dict):
        """
        Assumes the model is trained with a 'root' cell that shows up in every batch.
        Also assume the root is in the stem cell category 0.

        Note this requires data registry to contain `_indices`.
        """
        _indices = tensors["_indices"]
        _train_with_root = False
        start_ix = 0
        if self.iroot is not None and _indices[0].item() == self.iroot:
            _train_with_root = True
            start_ix = 1

        x = tensors[_CONSTANTS.X_KEY]
        batch_index = tensors[_CONSTANTS.BATCH_KEY]
        cont_covs = tensors.get(_CONSTANTS.CONT_COV_KEY, None)
        cat_covs = tensors.get(_CONSTANTS.CAT_COV_KEY, None)
        cat_groups = tensors[self.CAT_KEY]

        inference_outputs = self.inference(
            x=x, batch_index=batch_index, cont_covs=cont_covs, cat_covs=cat_covs
        )
        z = inference_outputs["z_sample"]
        T = (
            self.t_encoder(
                x[start_ix:],
                batch_index=batch_index[start_ix:],
                cont_covs=cont_covs[start_ix:] if cont_covs is not None else None,
                cat_covs=cat_covs[start_ix:] if cat_covs is not None else None,
            )
            .double()
            .squeeze()
        )
        if _train_with_root:
            T = torch.cat((torch.tensor([0.0]).to(x.device), T))

        sort_index, index = unique_index(T)
        T, z_ = T[sort_index][index], z[sort_index][index]
        # print(T.cpu().numpy()[:1000])

        # z_ = z.expand(self.n_cats, m, self.n_latent)
        # T_ = T.expand(self.n_cats, m, self.n_latent)
        pred_z = torch.zeros(len(index), self.n_latent).to(x.device)
        cat0_msk = (cat_groups == 0).squeeze()
        cat0_msk = cat0_msk[sort_index][index]
        if _train_with_root:
            z0 = z[0]
        self.ODEfuncs.to("cpu")
        for cat in range(self.n_cats):
            msk = (cat_groups == cat).squeeze()
            msk = msk[sort_index][index]
            T_msk = T[msk]
            # Tmin = torch.clone(T_msk).detach()[0]
            T_msk -= T[msk][0]  # if using root, then just substracts 0

            if cat == 0:
                if not _train_with_root:
                    z0 = z_[cat0_msk][0]
                T[msk] = T_msk
            else:
                z0 = pred_z[cat0_msk][-1]
                T0 = T[cat0_msk][-1]
                T[msk] = T_msk + T0

            options = get_step_size(self.step_size, T_msk[0], T_msk[-1], len(T_msk))
            pred_z_cat = (
                odeint(
                    self.ODEfuncs[cat],
                    z0.cpu(),
                    T_msk.cpu(),
                    method="euler",
                    options=options,
                )
                .view(-1, self.n_latent)  # minibatch, n_latent
                .to(x.device)
            )  # first element of pred_z_cat for cat 0 should be the root.
            pred_z[msk] = pred_z_cat

        generative_outputs = self.generative(
            z=z, batch_index=batch_index, cont_covs=cont_covs, cat_covs=cat_covs
        )
        outputs = dict()
        outputs.update(inference_outputs)
        outputs.update(generative_outputs)
        outputs.update(
            dict(
                pred_z=pred_z,
                T=T,
                index=index,
                sort_index=sort_index,
                pz_scale=self.pz_scale,
            )
        )

        return outputs

    def loss(self, tensors: dict):
        x = tensors[_CONSTANTS.X_KEY]
        outputs = self.forward(tensors)
        index = outputs["index"]
        sort_index = outputs["sort_index"]
        lib = torch.sum(x, dim=1).unsqueeze(1)
        x_rate = outputs["x_rate"]
        x_log_disp = outputs["x_log_disp"]
        pred_z = outputs["pred_z"]  # minibatch, n_latent
        z_mean = outputs["z_mean"][sort_index][index]
        z_var = outputs["z_var"][sort_index][index]

        lib = lib[sort_index][index]
        x_rate = x_rate[sort_index][index]
        x = x[sort_index][index]
        if x_log_disp.ndim > 1:
            x_log_disp = x_log_disp[sort_index][index]

        recon_loss = -NB(mu=lib * x_rate, theta=torch.exp(x_log_disp)).log_prob(x).sum(-1)

        kl_z = kl_divergence(
            Normal(z_mean, z_var.sqrt()),
            Normal(pred_z, torch.exp(self.pz_scale)),
        ).sum(-1)

        loss = torch.mean(recon_loss + self.kl_scaling * kl_z)

        return dict(
            loss=loss,
            kl_local=kl_z,
            reconstruction_error=recon_loss,
            kl_global=torch.tensor(0.0),
        )

    @torch.no_grad()
    def _get_pred_z(self, outputs: dict):
        """
        Depending on subclasses, pred_z can take a complicated multi-dimensional form,
        e.g. when using a categorical model and each category has its own pred_z.

        So use this function to wrap pred_z into one cells x n_latent matrix.
        """
        pred_z = outputs["pred_z"]
        return pred_z
