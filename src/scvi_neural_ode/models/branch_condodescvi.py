from typing import Optional

import numpy as np
import torch
from anndata import AnnData

from scvi_neural_ode.data import register_tensor_from_anndata
from scvi_neural_ode.models import CondODESCVI
from scvi_neural_ode.modules import BranchingODECVAE


class BranchingCondODESCVI(CondODESCVI):
    """
    Branching model using neural ODEs in latent space.

    Assumes one root population that splits off into multiple branches. We assume the root
    population is already labeled, and the branches are either known or can be inferred.

    The key model assumption is that all branch ODEs have the same starting point, defined as the
    last point in the root trajectory.

    Input
    ------
    n_cats
        Number of categories, including the root population.
    CAT_KEY
        Column (dtype int) to identify the root and branch populations, with 0 indicating the root.
    kl_scaling
        Multiplicative scale term in the KL divergence between latent variational posterior and generative distribution.
    iroot
        Index of root cell. If not None, model assumes it is starting point for trajectory of root population.
    """

    def __init__(
        self,
        adata: AnnData,
        n_cats: int,
        CAT_KEY: str,
        kl_scaling: float = 1.0,
        iroot: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(adata, **kwargs)
        self.adata.obs["_indices"] = np.arange(self.adata.n_obs)
        register_tensor_from_anndata(self.adata, CAT_KEY, "obs", CAT_KEY)
        register_tensor_from_anndata(self.adata, "_indices", "obs", "_indices")
        _module_kwargs = dict(
            n_input=self.adata.n_vars,
            n_batch=self.module.n_batch,
            n_cont_cov=self.module.n_cont_cov,
            n_cats_per_cov=self.module.n_cats_per_cov,
            kl_scaling=kl_scaling,
            CAT_KEY=CAT_KEY,
            n_cats=n_cats,
            iroot=iroot,
            **kwargs,
        )
        self.CAT_KEY = CAT_KEY

        self.module = BranchingODECVAE(**_module_kwargs)
        # iroot needs to be passed to the module AND the dataloader when calling train()

    def train(self, *args, **kwargs):
        if self.module.iroot is None:
            return super().train(*args, **kwargs)

        kwargs = dict() if kwargs is None else kwargs
        _update_dict = {"indices_in_every_batch": [self.module.iroot]}
        if "data_module_kwargs" in kwargs:
            kwargs["data_module_kwargs"].update(_update_dict)
        else:
            kwargs["data_module_kwargs"] = _update_dict

        return super().train(*args, **kwargs)

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
        outputs.update({"pred_z": self.module._get_pred_z(outputs=outputs)})

        for key in outputs.keys():
            outputs.update({key: outputs[key].cpu().numpy()})

        return outputs

    def get_time(self, *args):
        raise NotImplementedError(
            "This model rescales pseudotime so cannot call the time encoder. Instead call 'forward_pass'"
        )

    @torch.no_grad()
    def get_branch_categories(self):
        return self.adata.obs_vector(self.CAT_KEY)
