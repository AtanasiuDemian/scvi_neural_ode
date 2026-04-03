from typing import Iterable, Literal, Optional

import torch
import torch.nn as nn
from torch.distributions import Normal, Poisson, kl_divergence

from scvi_neural_ode import _CONSTANTS
from scvi_neural_ode.distributions import NB
from scvi_neural_ode.nn import Decoder, Encoder
from scvi_neural_ode.nn.utils import one_hot

from ._base import BaseModule


class VAE(BaseModule):
    def __init__(
        self,
        n_input,
        n_latent: int = 15,
        n_layers_encoder: int = 1,
        n_layers_decoder: int = 1,
        n_hidden: int = 128,
        dropout_rate: float = 0.1,
        n_batch: int = 0,
        n_cont_cov: int = 0,
        n_cats_per_cov: Optional[Iterable[int]] = None,
        use_batch_norm: Literal["encoder", "decoder", "none", "both"] = "both",
        use_layer_norm: Literal["encoder", "decoder", "none", "both"] = "none",
        gene_distribution: str = "NB",
        dispersion: Literal["gene", "batch-gene"] = "gene",
        log_variational: bool = False,
        beta_reg: float = 1.0,
    ):
        super().__init__()
        self.n_input = n_input
        self.n_latent = n_latent
        self.n_hidden = n_hidden
        self.n_layers_encoder = n_layers_encoder
        self.n_layers_decoder = n_layers_decoder
        # self.n_labels = n_batch
        self.n_batch = n_batch
        self.gene_distribution = gene_distribution
        self.log_variational = log_variational
        self.beta_reg = beta_reg
        self.mu, self.theta, self.nb_r, self.nb_p = True, True, False, False

        use_batch_norm_encoder = use_batch_norm == "encoder" or use_batch_norm == "both"
        use_batch_norm_decoder = use_batch_norm == "decoder" or use_batch_norm == "both"
        use_layer_norm_encoder = use_layer_norm == "encoder" or use_layer_norm == "both"
        use_layer_norm_decoder = use_layer_norm == "decoder" or use_layer_norm == "both"

        self.n_cont_cov = n_cont_cov
        self.n_cats_per_cov = n_cats_per_cov
        cat_list = [n_batch] + list([] if n_cats_per_cov is None else n_cats_per_cov)
        n_input_encoder = self.n_input + n_cont_cov
        self.encoder = Encoder(
            n_input_encoder,
            self.n_latent,
            n_cat_list=cat_list,
            n_layers=n_layers_encoder,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            use_batch_norm=use_batch_norm_encoder,
            use_layer_norm=use_layer_norm_encoder,
        )
        n_input_decoder = self.n_latent + n_cont_cov
        self.decoder = Decoder(
            n_input_decoder,
            self.n_input,
            n_cat_list=cat_list,
            n_layers=n_layers_decoder,
            n_hidden=n_hidden,
            use_batch_norm=use_batch_norm_decoder,
            use_layer_norm=use_layer_norm_decoder,
        )

        self.dispersion = dispersion
        if dispersion == "batch-gene":
            self.x_log_disp = nn.Parameter(torch.randn(self.n_input, self.n_batch))
        elif dispersion == "gene":
            self.x_log_disp = nn.Parameter(torch.randn(self.n_input))  # Gene specific dispersion.
        elif dispersion == "cat-gene":
            assert (
                n_cats_per_cov is not None
            ), "Can only use cat-gene if there extra categorical covariates in data registry."
            self.extra_n_cats = n_cats_per_cov[0]
            self.x_log_disp = nn.Parameter(torch.randn(self.n_input, self.extra_n_cats))
        else:
            raise ValueError(
                f"{dispersion} is not a valid dispersion form, choose from: `gene`, `batch-gene`, `cat-gene`."
            )
        self.gamma_shape = nn.Parameter(torch.randn(n_batch))
        self.gamma_scale = nn.Parameter(torch.randn(1))

    def inference(self, x, batch_index=None, cont_covs=None, cat_covs=None):
        x_ = x if not self.log_variational else torch.log(1 + x)

        if cont_covs is not None:
            encoder_input = torch.cat([x_, cont_covs], dim=-1)
        else:
            encoder_input = x_

        if cat_covs is not None:
            categorical_input = torch.split(cat_covs, 1, dim=1)
        else:
            categorical_input = tuple()

        z_mean, z_var, z_sample = self.encoder(encoder_input, batch_index, *categorical_input)

        return dict(z_mean=z_mean, z_var=z_var, z_sample=z_sample)

    def generative(self, z, batch_index, cont_covs=None, cat_covs=None):
        if cont_covs is not None:
            decoder_input = torch.cat([z, cont_covs], dim=-1)
        else:
            decoder_input = z

        if cat_covs is not None:
            categorical_input = torch.split(cat_covs, 1, dim=1)
        else:
            categorical_input = tuple()

        x_rate = self.decoder(decoder_input, batch_index, *categorical_input)  # gene proportions

        if self.dispersion == "gene":
            x_log_disp = self.x_log_disp
        elif self.dispersion == "batch-gene":
            x_log_disp = nn.functional.linear(one_hot(batch_index, self.n_batch), self.x_log_disp)
        elif self.dispersion == "cat-gene":
            x_log_disp = nn.functional.linear(one_hot(cat_covs, self.extra_n_cats), self.x_log_disp)

        return dict(x_rate=x_rate, x_log_disp=x_log_disp)

    def forward(self, tensors: dict, use_mean: bool = False):
        x = tensors[_CONSTANTS.X_KEY]
        batch = tensors[_CONSTANTS.BATCH_KEY]

        cont_covs = tensors.get(_CONSTANTS.CONT_COV_KEY, None)
        cat_covs = tensors.get(_CONSTANTS.CAT_COV_KEY, None)

        library = torch.sum(x, dim=1).unsqueeze(1)
        inference_outputs = self.inference(
            x, batch_index=batch, cont_covs=cont_covs, cat_covs=cat_covs
        )
        z_sample = (
            inference_outputs["z_sample"] if use_mean is False else inference_outputs["z_mean"]
        )
        generative_outputs = self.generative(
            z_sample, batch_index=batch, cont_covs=cont_covs, cat_covs=cat_covs
        )
        outputs = {}
        outputs.update(inference_outputs)
        outputs.update(generative_outputs)
        outputs["library"] = library

        return outputs

    def loss(self, tensors: dict, kl_weight: float = 1.0):
        x = tensors[_CONSTANTS.X_KEY]
        outputs = self.forward(tensors)
        library = outputs["library"]
        x_rate = outputs["x_rate"]
        x_log_disp = outputs["x_log_disp"]
        reconst_loss = self.reconstruction_loss(x, library, x_rate, x_log_disp)
        z_mean = outputs["z_mean"]
        z_var = outputs["z_var"]
        z_prior_mean = torch.zeros_like(z_mean)
        z_prior_scale = torch.ones_like(z_mean)
        kl_local = kl_divergence(
            Normal(z_mean, z_var.sqrt()), Normal(z_prior_mean, z_prior_scale)
        ).sum(dim=1)

        loss = torch.mean(reconst_loss + kl_weight * self.beta_reg * kl_local)

        return dict(
            loss=loss,
            kl_local=kl_local,
            reconstruction_error=reconst_loss,
            kl_global=torch.tensor(0.0),
        )

    def reconstruction_loss(self, x, library_size, x_rate, x_log_disp):
        if self.gene_distribution == "NB":
            if self.theta:
                _kwargs = {"theta": torch.exp(x_log_disp)}
            elif self.nb_p:
                _kwargs = {"logits": x_log_disp}
            return -NB(mu=library_size * x_rate, **_kwargs).log_prob(x).sum(dim=-1)
        elif self.gene_distribution == "Poisson":
            return -Poisson(library_size * x_rate + 1e-8).log_prob(x).sum(dim=-1)
        else:
            pass

    def sample_posterior_z(self, x, batch_index=None, use_mean=False):
        z_mean, _, z_sample = self.encoder(x, batch_index)
        if use_mean:
            return z_mean

        return z_sample
