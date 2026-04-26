from inspect import getfullargspec
from typing import Literal, Optional, Union

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

from scvi_neural_ode import _CONSTANTS
from scvi_neural_ode.nn import Classifier, one_hot


class TrainingPlan(pl.LightningModule):
    def __init__(
        self,
        module,
        lr: float = 1e-3,
        weight_decay: float = 1e-6,
        eps: float = 0.01,
        n_epochs_kl_warmup: Optional[int] = 100,
        reduce_lr_on_plateau: bool = True,
        metric_to_monitor="val_loss",
        lr_factor: float = 0.1,
        lr_patience: float = 20,
        lr_threshold: float = 1e-4,
        lr_min: float = 0,
        **loss_kwargs,
    ):
        super().__init__()
        self.module = module
        self.lr = lr
        self.weight_decay = weight_decay
        self.eps = eps
        self.reduce_lr_on_plateau = reduce_lr_on_plateau
        self.n_epochs_kl_warmup = n_epochs_kl_warmup
        self.metric_to_monitor = metric_to_monitor
        self.lr_factor = lr_factor
        self.lr_patience = lr_patience
        self.lr_threshold = lr_threshold
        self.lr_min = lr_min
        self.loss_kwargs = loss_kwargs

        # Pass KL weight only if required by the module loss function.
        self._loss_args = getfullargspec(self.module.loss)[0]

    def forward(self, *args, **kwargs):
        return self.module.forward(*args, **kwargs)

    def training_step(self, batch, batch_idx):
        if "kl_weight" in self._loss_args:
            self.loss_kwargs.update({"kl_weight": self.kl_weight})
        loss_dict = self.module.loss(batch, **self.loss_kwargs)
        loss = loss_dict["loss"]
        self.log("train_loss", loss, on_epoch=True, on_step=True, prog_bar=True, logger=True)
        return {
            "loss": loss,
            "kl_local": loss_dict["kl_local"].sum().detach(),
            "reconstruction_error": loss_dict["reconstruction_error"].sum().detach(),
            "kl_global": loss_dict["kl_global"].detach(),
            "n_obs": loss_dict["kl_local"].shape[0],
        }

    def validation_step(self, batch, batch_idx):
        loss_dict = self.module.loss(batch, **self.loss_kwargs)
        loss = loss_dict["loss"]
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        return {
            "val_loss": loss,
            "kl_local": loss_dict["kl_local"].sum().detach(),
            "reconstruction_error": loss_dict["reconstruction_error"].sum().detach(),
            "kl_global": loss_dict["kl_global"].detach(),
            "n_obs": loss_dict["kl_local"].shape[0],
        }

    def on_train_epoch_end(self):
        pass

    def on_validation_epoch_end(self):
        pass

    def configure_optimizers(self):
        params = filter(lambda p: p.requires_grad, self.module.parameters())
        optimizer = torch.optim.AdamW(
            params,
            lr=self.lr,
            weight_decay=self.weight_decay,
            eps=self.eps,
        )
        config = {"optimizer": optimizer}
        if self.reduce_lr_on_plateau is True:
            scheduler = ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=self.lr_factor,
                patience=self.lr_patience,
                threshold=self.lr_threshold,
            )
            config.update({"lr_scheduler": scheduler, "monitor": self.metric_to_monitor})

        return config

    @property
    def kl_weight(self):
        if self.n_epochs_kl_warmup is not None:
            kl_weight = min(1.0, self.current_epoch / self.n_epochs_kl_warmup)
        else:
            kl_weight = 1.0

        return kl_weight


class AdversarialTrainingPlan(TrainingPlan):
    def __init__(
        self,
        module,
        lr: float = 1e-3,
        weight_decay: float = 1e-6,
        n_epochs_kl_warmup: Optional[int] = 100,
        reduce_lr_on_plateau: bool = True,
        lr_factor: float = 0.1,
        lr_patience: int = 30,
        lr_threshold: float = 0.0,
        metric_to_monitor="val_loss",
        lr_min: float = 0,
        adversarial_classifier: Union[bool, Classifier] = True,
        scale_adversarial_loss: Union[float, Literal["auto"]] = "auto",
    ):
        super().__init__(
            module=module,
            lr=lr,
            weight_decay=weight_decay,
            n_epochs_kl_warmup=n_epochs_kl_warmup,
            reduce_lr_on_plateau=reduce_lr_on_plateau,
            lr_factor=lr_factor,
            lr_patience=lr_patience,
            lr_threshold=lr_threshold,
            lr_min=lr_min,
            metric_to_monitor=metric_to_monitor,
        )
        if adversarial_classifier is True:
            self.n_labels_classifier = self.module.n_batch
            self.classifier = Classifier(
                n_input=self.module.n_latent,
                n_hidden=32,
                n_labels=self.n_labels_classifier,
                n_layers=2,
                use_logits=True,
            )
        else:
            self.classifier = adversarial_classifier
        self.scale_adversarial_loss = scale_adversarial_loss
        self.automatic_optimization = False

    def classifier_loss(self, z, batch_index, predict_true_class=True):
        n_classes = self.n_labels_classifier
        cls_logits = nn.LogSoftmax(dim=1)(self.classifier(z))

        if predict_true_class is True:
            cls_target = one_hot(batch_index, n_classes)
        else:
            one_hot_batch = one_hot(batch_index, n_classes)
            cls_target = (~one_hot_batch.bool()).float()
            cls_target = cls_target / (n_classes - 1)

        l_soft = cls_logits * cls_target
        loss = -l_soft.sum(1).mean()

        return loss

    def training_step(self, batch, batch_idx):
        opt1, opt2 = self.optimizers()

        if "kl_weight" in self._loss_args:
            self.loss_kwargs.update({"kl_weight": self.kl_weight})

        kappa = (
            1 - self.kl_weight
            if self.scale_adversarial_loss == "auto"
            else self.scale_adversarial_loss
        )

        batch_tensor = batch[_CONSTANTS.BATCH_KEY]

        # =========================
        # 1. Main model update
        # =========================
        loss_dict = self.module.loss(batch, **self.loss_kwargs)
        loss = loss_dict["loss"]

        if kappa > 0 and self.classifier is not False:
            module_outputs = self.module.forward(batch)
            z = module_outputs["z_sample"]
            fool_loss = self.classifier_loss(z, batch_tensor, predict_true_class=False)
            loss = loss + kappa * fool_loss

        self.manual_backward(loss)
        opt1.step()
        opt1.zero_grad()

        # =========================
        # 2. Classifier update
        # =========================
        if self.classifier:
            module_outputs = self.module.forward(batch)
            z = module_outputs["z_sample"]

            cls_loss = self.classifier_loss(z.detach(), batch_tensor, predict_true_class=True)
            cls_loss = cls_loss * kappa

            self.manual_backward(cls_loss)
            opt2.step()
            opt2.zero_grad()

        # =========================
        # Logging
        # =========================
        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        return {
            "loss": loss.detach(),
            "kl_local": loss_dict["kl_local"].sum().detach(),
            "reconstruction_error": loss_dict["reconstruction_error"].sum().detach(),
            "kl_global": loss_dict["kl_global"].detach(),
            "n_obs": loss_dict["kl_local"].shape[0],
        }

    def configure_optimizers(self):
        # --- Optimizer 1 (main module) ---
        params1 = filter(lambda p: p.requires_grad, self.module.parameters())
        optimizer1 = torch.optim.AdamW(
            params1,
            lr=self.lr,
            weight_decay=self.weight_decay,
            eps=self.eps,
        )

        optimizers = [optimizer1]
        schedulers = []

        if self.reduce_lr_on_plateau:
            scheduler1 = {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer1,
                    mode="min",
                    factor=self.lr_factor,
                    patience=self.lr_patience,
                    threshold=self.lr_threshold,
                    min_lr=self.lr_min,
                ),
                "monitor": self.metric_to_monitor,
            }
            schedulers.append(scheduler1)

        # --- Optional second optimizer (classifier) ---
        if self.classifier:
            params2 = filter(lambda p: p.requires_grad, self.classifier.parameters())
            optimizer2 = torch.optim.Adam(
                params2,
                lr=1e-3,
                eps=0.01,
                weight_decay=self.weight_decay,
            )
            optimizers.append(optimizer2)

        # --- Return format ---
        if schedulers:
            return optimizers, schedulers
        return optimizers
