import torch
import torch.nn as nn
from torch.distributions import Normal, kl_divergence
from torchdiffeq import odeint

from scvi_neural_ode import _CONSTANTS
from scvi_neural_ode.distributions import NB
from scvi_neural_ode.modules import VAE
from scvi_neural_ode.nn import ODEfunc, TimeEncoder
from scvi_neural_ode.utils.ode import get_step_size, unique_index


class ODECVAE(VAE):
    """
    Conditional VAE with pseudotime dynamics parameterized by a neural ODE.

    Takes as input categorical and continuous covariates and infers a latent space with one trajectory.
    """

    def __init__(
        self,
        n_input: int,
        n_latent: int,
        step_size: int = None,
        kl_scaling: float = 1.0,
        condition_time_encoder: bool = True,
        **kwargs,
    ):
        super().__init__(n_input=n_input, n_latent=n_latent, **kwargs)
        self.kl_scaling = kl_scaling
        self.step_size = step_size
        self.pz_scale = nn.Parameter(torch.randn(self.n_latent))
        self.ODEfunc = ODEfunc(
            n_input=self.n_latent,
            n_hidden=64,
            n_output=self.n_latent,
            activation_fn=nn.ELU,
        )

        self.condition_time_encoder = condition_time_encoder
        n_cat_list = None
        if condition_time_encoder:
            n_cat_list = [self.n_batch] + list(
                [] if self.n_cats_per_cov is None else self.n_cats_per_cov
            )
        t_encoder_input = self.n_input + self.n_cont_cov * condition_time_encoder
        self.t_encoder = TimeEncoder(
            n_input=t_encoder_input, n_hidden=self.n_hidden, n_cat_list=n_cat_list
        )

    def solve_ode(self, func, z0, T, method="euler"):
        options = get_step_size(self.step_size, T[0], T[-1], len(T))
        pred_z = (
            odeint(
                func.cpu(),
                z0.cpu(),
                T.cpu(),
                method=method,
                options=options,
            )
            .view(-1, self.n_latent)  # minibatch, n_latent
            .to(T.device)
        )

        return pred_z

    def forward(self, tensors: dict):
        """
        In addition to the forward pass outputs of the parent class, returns

        pred_z
            Latents corresponding to the solution of the neural ODE.
            The cells indexed by this tensor are SORTED in pseudotime - unlike `z_sample`.
        sort_index, index
            Indices that sort cells and filter for unique values by pseudotime values.
        T
            Sorted values of pseudotime.
        """
        x = tensors[_CONSTANTS.X_KEY]
        batch = tensors[_CONSTANTS.BATCH_KEY]
        cont_covs = tensors.get(_CONSTANTS.CONT_COV_KEY, None)
        cat_covs = tensors.get(_CONSTANTS.CAT_COV_KEY, None)
        inference_outputs = self.inference(
            x=x, batch_index=batch, cont_covs=cont_covs, cat_covs=cat_covs
        )
        z = inference_outputs["z_sample"]
        t_encoder_kwargs = dict()
        if self.condition_time_encoder:
            t_encoder_kwargs.update(dict(batch_index=batch, cont_covs=cont_covs, cat_covs=cat_covs))
        T = self.t_encoder(x, **t_encoder_kwargs).double().ravel()
        sort_index, index = unique_index(T)
        T, z_ = T[sort_index][index], z[sort_index][index]

        self.ODEfunc.to("cpu")
        pred_z = torch.zeros_like(z_).to(x.device)
        z0 = z_[0]
        options = get_step_size(self.step_size, T[0], T[-1], len(T))
        pred_z = (
            odeint(
                self.ODEfunc,
                z0.cpu(),
                T.cpu(),
                method="euler",
                options=options,
            )
            .view(-1, self.n_latent)
            .to(x.device)
        )
        generative_outputs = self.generative(
            z=z, batch_index=batch, cont_covs=cont_covs, cat_covs=cat_covs
        )
        outputs = dict()
        outputs.update(inference_outputs)
        outputs.update(generative_outputs)
        outputs.update(dict(pred_z=pred_z, T=T, index=index, sort_index=sort_index))

        return outputs

    def loss(self, tensors: dict):
        x = tensors[_CONSTANTS.X_KEY]
        outputs = self.forward(tensors)
        lib = torch.sum(x, dim=1).unsqueeze(1)
        x_rate = outputs["x_rate"]
        x_log_disp = outputs["x_log_disp"]
        pred_z = outputs["pred_z"]
        sort_index = outputs["sort_index"]
        index = outputs["index"]
        z_mean, z_var = (
            outputs["z_mean"][sort_index][index],
            outputs["z_var"][sort_index][index],
        )
        x = x[sort_index][index]
        lib = lib[sort_index][index]
        x_rate = x_rate[sort_index][index]
        if self.x_log_disp.ndim > 1:
            x_log_disp = x_log_disp[sort_index][index]

        recon_loss = -NB(mu=lib * x_rate, theta=torch.exp(x_log_disp)).log_prob(x).sum(-1)
        z_scale = torch.nn.functional.softplus(self.pz_scale)
        kl_z = kl_divergence(Normal(z_mean, z_var.sqrt()), Normal(pred_z, z_scale)).sum(-1)

        loss = torch.mean(recon_loss + self.kl_scaling * kl_z)

        return dict(
            loss=loss,
            kl_local=kl_z,
            reconstruction_error=recon_loss,
            kl_global=torch.tensor(0.0),
        )
