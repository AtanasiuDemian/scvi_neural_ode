from typing import Optional

import torch
import torch.nn as nn
from torch.distributions import Normal, kl_divergence
from torchdiffeq import odeint

from scvi_neural_ode import _CONSTANTS
from scvi_neural_ode.distributions import NB
from scvi_neural_ode.modules import VAE
from scvi_neural_ode.nn import ODEfunc, TimeEncoder
from scvi_neural_ode.utils.ode import get_step_size, unique_index


class ODEVAE(VAE):
    """
    VAE with pseudotime dynamics parameterized by a neural ODE.

    Here each cell group described by the batch covariate has its own trajectory.
    This is suitable for inferring individual trajectories of unrelated cell types.
    """

    def __init__(
        self,
        n_input: int,
        n_latent: int,
        step_size: int = None,
        kl_scaling: float = 1.0,
        time_encoder_params: Optional[dict] = None,
        solver_method: str = "dopri5",
        **kwargs,
    ):
        super().__init__(
            n_input=n_input,
            n_latent=n_latent,
            **kwargs,
        )
        self.kl_scaling = kl_scaling
        self.step_size = step_size
        self.pz_scale = nn.Parameter(torch.randn(self.n_latent))
        self.ODEfunc = nn.ModuleList(
            [
                ODEfunc(
                    n_input=self.n_latent,
                    n_hidden=64,
                    n_output=self.n_latent,
                    activation_fn=nn.ELU,
                )
                for _ in range(self.n_batch)
            ]
        )
        self.solver_method = solver_method

        time_encoder_params = {} if time_encoder_params is None else time_encoder_params
        self.t_encoder = TimeEncoder(
            n_input=n_input, n_cat_list=[self.n_batch], **time_encoder_params
        )

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
        self.ODEfunc.to("cpu")
        outputs = {}
        for label_index in range(self.n_batch):
            outputs[label_index] = {}
            x = tensors[label_index][_CONSTANTS.X_KEY]
            batch_tensor = tensors[label_index][_CONSTANTS.BATCH_KEY]
            # ^ this is a tensor of one unique value `batch_index`

            inference_outputs = self.inference(x, batch_index=batch_tensor)
            z = inference_outputs["z_sample"]
            T = self.t_encoder(x, batch_index=batch_tensor).double().ravel()
            sort_index, index = unique_index(T)
            T, z_ = T[sort_index][index], z[sort_index][index]
            if T.shape[0] < 5:
                continue
            z0 = z_[0]
            options = get_step_size(self.step_size, T[0], T[-1], len(T))
            # print(z0, T)
            pred_z = (
                odeint(
                    self.ODEfunc[label_index],
                    z0.cpu(),
                    T.cpu(),
                    method=self.solver_method,
                    options=options,
                )
                .view(-1, self.n_latent)
                .to(x.device)
            )
            generative_outputs = self.generative(z, batch_tensor)
            outputs[label_index].update(inference_outputs)
            outputs[label_index].update(
                dict(pred_z=pred_z, T=T, index=index, sort_index=sort_index)
            )
            outputs[label_index].update(generative_outputs)

        return outputs

    def loss(self, tensors: dict):
        outputs = self.forward(tensors)
        total_recon_loss, total_kl_z = [], []
        for label_index in range(self.n_batch):
            x = tensors[label_index][_CONSTANTS.X_KEY]
            lib = x.sum(1).unsqueeze(1)

            sort_index = outputs[label_index]["sort_index"]
            index = outputs[label_index]["index"]
            x_rate = outputs[label_index]["x_rate"][sort_index][index]
            z_mean = outputs[label_index]["z_mean"][sort_index][index]
            z_var = outputs[label_index]["z_var"][sort_index][index]
            x = x[sort_index][index]
            lib = lib[sort_index][index]
            x_log_disp = outputs[label_index]["x_log_disp"]
            if x_log_disp.ndim > 1:
                x_log_disp = x_log_disp[sort_index][index]

            total_recon_loss += [
                -NB(mu=lib * x_rate, theta=torch.exp(x_log_disp)).log_prob(x).sum(-1)
            ]

            pred_z = outputs[label_index]["pred_z"]
            total_kl_z += [
                kl_divergence(
                    Normal(z_mean, z_var.sqrt()),
                    Normal(pred_z, torch.exp(self.pz_scale)),
                ).sum(-1)
            ]

        total_recon_loss = torch.cat(total_recon_loss)
        total_kl_z = torch.cat(total_kl_z)

        loss = torch.mean(total_recon_loss + self.kl_scaling * total_kl_z)

        return dict(
            loss=loss,
            kl_local=total_kl_z,
            reconstruction_error=total_recon_loss,
            kl_global=torch.tensor(0.0),
        )

    @torch.no_grad()
    def get_time(self, x: torch.Tensor, label_tensor: torch.Tensor):
        """Return pseudotime values from time encoder output."""
        return self.t_encoder(x, label_tensor).ravel().cpu().numpy()

    @torch.no_grad()
    def get_trajectory_output(self, tensors):
        """
        Returns trajectory information for a given label.

        This consists of:
            pseudotime values (T),
            predicted latent trajectory (pred_z)
            latents sampled from variational posterior sorted according to T (z_mean, z_var)
            predicted gene expression corresponding to the latent trajectory (pred_x)
            sorted cell indices according to T (if indices is not None)
        """
        output = {}
        forward_outputs = self.forward(tensors)
        for label_index in range(self.n_batch):
            sort_index = forward_outputs[label_index]["sort_index"]
            index = forward_outputs[label_index]["index"]
            pred_z = forward_outputs[label_index]["pred_z"]
            T = forward_outputs[label_index]["T"]
            z_mean = forward_outputs[label_index]["z_mean"]
            z_var = forward_outputs[label_index]["z_var"]
            z_sample = forward_outputs[label_index]["z_sample"]

            z_mean = z_mean[sort_index][index]
            z_var = z_var[sort_index][index]
            z_sample = z_sample[sort_index][index]

            label_tensor = label_index * torch.ones((pred_z.shape[0], 1)).to(pred_z.device)
            pred_x = self.decoder(pred_z, label_tensor)

            output[label_index] = dict(
                T=T,
                pred_z=pred_z,
                pred_x=pred_x,
                z_mean=z_mean,
                z_var=z_var,
                z_sample=z_sample,
                sort_index=sort_index,
                index=index,
            )

        return output

    @torch.no_grad()
    def forward_pass(self, x: torch.Tensor, label_tensor: torch.Tensor, use_mean: bool = False):
        """Forward pass without calling the ODE solver."""
        output_dict = {}
        inference_outputs = self.inference(x=x, batch_index=label_tensor)
        z = inference_outputs["z_mean"] if use_mean else inference_outputs["z_sample"]
        generative_outputs = self.generative(z=z, batch_index=label_tensor)
        output_dict.update(inference_outputs)
        output_dict.update(generative_outputs)

        return output_dict
