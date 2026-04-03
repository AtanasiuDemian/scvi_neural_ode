from typing import Optional

import torch
from anndata import AnnData

from scvi_neural_ode import _CONSTANTS
from scvi_neural_ode.models import SCVI
from scvi_neural_ode.modules import ODECVAE


class CondODESCVI(SCVI):
    """
    Conditional VAE with pseudotime dynamics parameterized by a neural ODE in the latent space.

    Takes as input categorical and continuous covariates and infers a latent space with one trajectory.
    """

    def __init__(
        self,
        adata,
        kl_scaling=1.0,
        use_adversarial_classifier: bool = False,
        **module_kwargs,
    ):
        super().__init__(
            adata,
            use_adversarial_classifier=use_adversarial_classifier,
        )
        self.module = ODECVAE(
            n_input=self.adata.n_vars,
            n_batch=self.module.n_batch,
            n_cont_cov=self.module.n_cont_cov,
            n_cats_per_cov=self.module.n_cats_per_cov,
            kl_scaling=kl_scaling,
            **module_kwargs,
        )

    @torch.no_grad()
    def get_time(self, adata: Optional[AnnData] = None):
        if adata is None:
            adata = self.adata
        scdl = self._make_data_loader(adata, shuffle=False)
        output = []
        for tensors in scdl:
            x = tensors[_CONSTANTS.X_KEY]
            batch_index = tensors[_CONSTANTS.BATCH_KEY]
            cont_covs = tensors.get(_CONSTANTS.CONT_COV_KEY, None)
            cat_covs = tensors.get(_CONSTANTS.CAT_COV_KEY, None)
            _kwargs = {}
            if self.module.condition_time_encoder:
                _kwargs.update(
                    dict(batch_index=batch_index, cont_covs=cont_covs, cat_covs=cat_covs)
                )
            t_ = self.module.t_encoder(x, **_kwargs)
            output += [t_]

        return torch.cat(output).cpu().numpy().squeeze()

    @torch.no_grad()
    def forward_pass(self, adata: Optional[AnnData] = None):
        """
        Runs one forward pass over the full dataset.

        Difficult to run this in minibatches as each output would be sorted.
        """
        if adata is None:
            adata = self.adata
        scdl = self._make_data_loader(adata, batch_size=adata.n_obs, shuffle=False)
        tensors = [tensors for tensors in scdl][0]
        outputs = self.module(tensors)

        for key in outputs.keys():
            outputs.update({key: outputs[key].cpu().numpy()})

        return outputs
