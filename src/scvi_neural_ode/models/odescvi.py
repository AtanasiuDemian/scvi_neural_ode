from typing import List, Optional, Union

import numpy as np
import pandas as pd
import torch
from anndata import AnnData
from lightning.pytorch.accelerators import Accelerator

from scvi_neural_ode import _CONSTANTS
from scvi_neural_ode.data import (
    AnnDataLoader,
    ConcatAnnDataLoader,
    ConditionalAnnDataModule,
)
from scvi_neural_ode.models import SCVI
from scvi_neural_ode.modules import ODEVAE


class ODESCVI(SCVI):
    """
    VAE with pseudotime dynamics parameterized by a neural ODE in the latent space.

    Here each cell group described by the `batch` covariate has its own trajectory.
    This is suitable for inferring individual trajectories of unrelated cell types.
    """

    def __init__(
        self,
        adata,
        kl_scaling=1.0,
        CATEGORY_KEY: Optional[str] = None,
        solver_method: str = "dopri5",
        **kwargs,
    ):
        super().__init__(adata, **kwargs)
        if self.module.n_cont_cov != 0 or self.module.n_cats_per_cov is not None:
            raise NotImplementedError

        self.module = ODEVAE(
            n_input=self.adata.n_vars,
            n_batch=self.module.n_batch,
            kl_scaling=kl_scaling,
            solver_method=solver_method,
            **kwargs,
        )
        self._data_module_cls = ConditionalAnnDataModule
        if CATEGORY_KEY is not None:
            self.mapping = pd.Categorical(adata.obs[CATEGORY_KEY]).categories.values
        else:
            self.mapping = np.arange(self.module.n_batch)

    def train(
        self,
        n_epochs: Optional[int] = 400,
        lr: Optional[float] = 1e-3,
        accelerator: Union[str, Accelerator] = "auto",
        train_size: float = 0.9,
        validation_size: Optional[float] = None,
        batch_size: int = 128,
        plan_kwargs: Optional[dict] = None,
        data_module_kwargs: Optional[dict] = None,
        **trainer_kwargs,
    ):
        data_module_kwargs = data_module_kwargs if isinstance(data_module_kwargs, dict) else dict()
        data_module_kwargs["key"] = _CONSTANTS.BATCH_KEY
        return super().train(
            n_epochs=n_epochs,
            lr=lr,
            accelerator=accelerator,
            train_size=train_size,
            validation_size=validation_size,
            batch_size=batch_size,
            plan_kwargs=plan_kwargs,
            data_module_kwargs=data_module_kwargs,
            **trainer_kwargs,
        )

    @torch.no_grad()
    def get_time(
        self,
        adata: Optional[AnnData] = None,
        indices: Optional[List] = None,
        batch_size=128,
    ):
        """Return pseudotime values from time encoder output."""
        adata = adata if adata is not None else self.adata
        scdl = AnnDataLoader(adata, indices=indices, batch_size=batch_size, shuffle=False)
        output = []
        for tensors in scdl:
            x = tensors[_CONSTANTS.X_KEY]
            label_tensor = tensors[_CONSTANTS.BATCH_KEY]
            output += [self.module.get_time(x, label_tensor)]

        return np.concatenate(output)

    @torch.no_grad()
    def get_trajectory_output(self):
        """
        Returns trajectory information for each label.

        This consists of:
            pseudotime values (T),
            predicted latent trajectory (pred_z)
            latents sampled from variational posterior sorted according to T (z_mean, z_var)
            predicted gene expression corresponding to the latent trajectory (pred_x)
            sorted cell indices according to T (if indices is not None)

        Output
        ------
        dict with keys given by labels. The values of this dict are dictionaries with the above keys.
        """
        # Don't need the below 2 lines unless you are minibatching per each label.
        # self.adata.obs["indices"] = np.arange(self.adata.n_obs)
        # register_tensor_from_anndata(self.adata, "indices", "obs", "indices")
        output = {}
        indices_list = [
            np.where(self.adata.obs_vector(_CONSTANTS.BATCH_KEY) == b)[0]
            for b in range(self.module.n_batch)
        ]
        dl = ConcatAnnDataLoader(
            adata=self.adata,
            indices_list=indices_list,
            shuffle=False,
            batch_size=self.adata.n_obs,
        )
        tensors = next(iter(dl))
        traj_output = self.module.get_trajectory_output(tensors)
        for label_index, label in enumerate(self.mapping):
            sort_index = traj_output[label_index]["sort_index"].cpu().numpy()
            index = traj_output[label_index]["index"].cpu().numpy()
            pred_z = traj_output[label_index]["pred_z"].cpu().numpy()
            pred_x = traj_output[label_index]["pred_x"].cpu().numpy()
            T = traj_output[label_index]["T"].cpu().numpy()
            z_mean = traj_output[label_index]["z_mean"].cpu().numpy()
            z_var = traj_output[label_index]["z_var"].cpu().numpy()
            z_sample = traj_output[label_index]["z_sample"].cpu().numpy()

            indices = indices_list[label_index][sort_index][index]
            pred_x = pd.DataFrame(pred_x, index=indices, columns=self.adata.var_names)

            indices = indices_list[label_index][sort_index][index]
            # would this create an issue with sort_index & index devices?
            output[label] = dict(
                T=T,
                pred_z=pred_z,
                pred_x=pred_x,
                z_sample=z_sample,
                z_mean=z_mean,
                z_var=z_var,
                indices=indices,
            )

        return output

    @torch.no_grad()
    def get_latent_representation(
        self,
        adata: Optional[AnnData] = None,
        indices: Optional[List[int]] = None,
        use_mean: bool = True,
        batch_size: int = 128,
    ):
        # Need to rewrite this because the parent calls module .forward method
        adata = adata if adata is not None else self.adata
        scdl = AnnDataLoader(adata=adata, indices=indices, batch_size=batch_size, shuffle=False)
        latents = []
        for tensors in scdl:
            X = tensors[_CONSTANTS.X_KEY]
            label_tensor = tensors[_CONSTANTS.BATCH_KEY]
            module_outputs = self.module.forward_pass(X, label_tensor)
            zs = module_outputs["z_mean"] if use_mean else module_outputs["z_sample"]
            latents += [zs.cpu().numpy()]

        return np.concatenate(latents, axis=0)

    @torch.no_grad()
    def get_normalized_expression(
        self,
        adata: Optional[AnnData] = None,
        indices: Optional[List[int]] = None,
        gene_list: Optional[List[str]] = None,
        size_factor: float = 1.0,
        batch_size: int = 128,
        return_numpy: bool = True,
        use_mean: bool = True,
    ):
        # Need to rewrite this because the parent calls module .forward method
        adata = adata if adata is not None else self.adata
        indices = indices if indices is not None else np.arange(adata.n_obs)
        if gene_list is None:
            gene_mask = slice(None)
        else:
            all_genes = adata.var_names.values
            gene_mask = np.array([True if gene in gene_list else False for gene in all_genes])

        exprs = []
        scdl = AnnDataLoader(adata=adata, indices=indices, batch_size=batch_size, shuffle=False)
        for tensors in scdl:
            X = tensors[_CONSTANTS.X_KEY]
            label_tensor = tensors[_CONSTANTS.BATCH_KEY]
            module_outputs = self.module.forward_pass(X, label_tensor, use_mean=use_mean)
            rate = module_outputs["x_rate"][..., gene_mask].cpu().numpy()
            rate *= size_factor
            exprs += [rate]
        exprs = np.concatenate(exprs, axis=0)

        if not return_numpy:
            return pd.DataFrame(
                exprs,
                columns=adata.var_names.values if gene_list is None else gene_list,
                index=adata.obs_names[indices],
            )
        return exprs
