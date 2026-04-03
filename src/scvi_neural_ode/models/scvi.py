from typing import Optional, Sequence, Union

import numpy as np
import torch
from anndata import AnnData
from lightning.pytorch.accelerators import Accelerator

from scvi_neural_ode import _CONSTANTS
from scvi_neural_ode.data import get_from_registry
from scvi_neural_ode.models.base import (
    BaseModelAbstractClass,
    DifferentialMixin,
    RNASeqMixin,
    TrainingMixin,
    VAEMixin,
)
from scvi_neural_ode.modules import VAE
from scvi_neural_ode.train import AdversarialTrainingPlan, Trainer, TrainingPlan


class SCVI(
    RNASeqMixin,
    VAEMixin,
    TrainingMixin,
    DifferentialMixin,
    BaseModelAbstractClass,
):
    def __init__(
        self,
        adata: AnnData,
        n_latent: int = 15,
        n_hidden: int = 128,
        n_layers_encoder: int = 1,
        n_layers_decoder: int = 1,
        dropout_rate: float = 0.1,
        batch_size: int = 128,
        use_adversarial_classifier: bool = False,
        **module_kwargs,
    ):
        super().__init__(adata, batch_size)
        registry_dict = adata.uns["data_registry"]
        if _CONSTANTS.BATCH_KEY in registry_dict.keys():
            n_batch = len(np.unique(get_from_registry(adata, _CONSTANTS.BATCH_KEY)))
        else:
            n_batch = 0

        if _CONSTANTS.CONT_COV_KEY in registry_dict.keys():
            _, attr_key = registry_dict[_CONSTANTS.CONT_COV_KEY]
            if isinstance(attr_key, list):
                n_cont_cov = len(attr_key)
            elif isinstance(attr_key, str):
                n_cont_cov = 1
        else:
            n_cont_cov = 0

        if _CONSTANTS.CAT_COV_KEY in registry_dict.keys():
            _, attr_key = registry_dict[_CONSTANTS.CAT_COV_KEY]
            data = get_from_registry(adata, _CONSTANTS.CAT_COV_KEY)
            # If attr_key is a list then data is a DataFrame, else if attr_key is str then data is ndarray.
            if isinstance(attr_key, list):
                n_cats_per_cov = [len(np.unique(data[k])) for k in attr_key]
            elif isinstance(attr_key, str):
                n_cats_per_cov = [len(np.unique(data))]
        else:
            n_cats_per_cov = None

        self.module = VAE(
            n_input=adata.n_vars,
            n_batch=n_batch,
            n_hidden=n_hidden,
            n_latent=n_latent,
            n_layers_encoder=n_layers_encoder,
            n_layers_decoder=n_layers_decoder,
            dropout_rate=dropout_rate,
            n_cont_cov=n_cont_cov,
            n_cats_per_cov=n_cats_per_cov,
            **module_kwargs,
        )
        self.use_adversarial_classifier = use_adversarial_classifier
        self.init_params_ = self._get_init_params(locals())

    @torch.no_grad()
    def get_vamp_prior(
        self,
        adata: Optional[AnnData] = None,
        p: int = 10,
    ):
        mean_vprior = np.zeros((self.module.n_batch, p, self.module.n_latent))
        var_vprior = np.zeros_like(mean_vprior)
        self.vamp_indices = []
        if adata is None:
            adata = self.adata
        # NOTE: this assumes the categories we sample the VAMP prior with respect to, are from BATCH_KEY
        assert (
            p <= np.unique(adata.obs[_CONSTANTS.BATCH_KEY], return_counts=True)[1].min()
        ), "Number of single cell samples in VAMP prior need to be smaller than the number of cells in each cluster"
        for ct in range(self.module.n_batch):
            mean, var = [], []
            # Each single cell sample is unique.
            _, attr_key = adata.uns["data_registry"][_CONSTANTS.BATCH_KEY]
            if isinstance(attr_key, list):
                raise NotImplementedError(
                    "`get_vamp_prior` only works if _CONSTANTS.BATCH_KEY has only one attribute key."
                )
            ct_indices = np.random.choice(
                np.where(adata.obs_vector(attr_key) == ct)[0], size=p, replace=False
            )
            self.vamp_indices.append(ct_indices)
            scdl = self._make_data_loader(
                adata=adata,
                indices=ct_indices,
                batch_size=p,
            )
            for tensors in scdl:
                x = tensors[_CONSTANTS.X_KEY]
                batch = tensors[_CONSTANTS.BATCH_KEY]
                cont_covs = tensors.get(_CONSTANTS.CONT_COV_KEY, None)
                cat_covs = tensors.get(_CONSTANTS.CAT_COV_KEY, None)
                inference_outputs = self.module.inference(
                    x=x, batch_index=batch, cont_covs=cont_covs, cat_covs=cat_covs
                )
                z_mean, z_var = inference_outputs["z_mean"], inference_outputs["z_var"]
                mean += [z_mean.cpu()]
                var += [z_var.cpu()]
            mean_vprior[ct] = np.array(torch.cat(mean))
            var_vprior[ct] = np.array(torch.cat(var))

        return mean_vprior, var_vprior

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
        enable_checkpointing: bool = False,
        **trainer_kwargs,
    ):
        # Make sure the updating of plan_kwargs makes sense and all args are passed through.
        plan_kwargs = plan_kwargs if isinstance(plan_kwargs, dict) else dict()
        if self.use_adversarial_classifier:
            _training_plan_cls = AdversarialTrainingPlan
            plan_kwargs.update({"adversarial_classifier": self.use_adversarial_classifier})
            if "scale_adversarial_loss" not in plan_kwargs:
                plan_kwargs.update({"scale_adversarial_loss": 1.0})
        else:
            _training_plan_cls = TrainingPlan

        self.training_input = {
            "n_epochs": n_epochs,
            "lr": lr,
            "train_size": train_size,
            "validation_size": validation_size,
            "batch_size": batch_size,
        }
        self.training_input.update(plan_kwargs)

        data_module_kwargs = data_module_kwargs if isinstance(data_module_kwargs, dict) else dict()
        self.training_input.update(data_module_kwargs)
        self.data_module = self._data_module_cls(
            self.adata,
            train_size=train_size,
            validation_size=validation_size,
            batch_size=batch_size,
            **data_module_kwargs,
        )
        self.data_module.setup()
        if self.data_module.train_size == 1:
            plan_kwargs.update({"metric_to_monitor": "train_loss"})

        training_plan = _training_plan_cls(self.module, lr, **plan_kwargs)
        trainer = Trainer(
            n_epochs=n_epochs, accelerator=accelerator, enable_checkpointing=enable_checkpointing, **trainer_kwargs
        )
        trainer.fit(model=training_plan, datamodule=self.data_module)
        self.train_indices = self.data_module.train_idx
        self.test_indices = self.data_module.test_idx
        self.validation_indices = self.data_module.val_idx
        self.module.eval()
        self.is_trained_ = True
        self.to_device(self.device)
        self.trainer = trainer

    @torch.no_grad()
    def posterior_z_params(
        self,
        adata: Optional[AnnData] = None,
        indices: Sequence[int] = None,
        batch_size: int = 128,
    ):
        if adata is None:
            adata = self.adata
        scdl = self._make_data_loader(adata, indices=indices, batch_size=batch_size, shuffle=False)
        z_mean, z_var = [], []
        for tensors in scdl:
            outputs = self.module.forward(tensors)
            z_mean += [outputs["z_mean"]]
            z_var += [outputs["z_var"]]

        z_mean = torch.cat(z_mean).cpu().numpy()
        z_var = torch.cat(z_var).cpu().numpy()

        return {"z_mean": z_mean, "z_var": z_var}
