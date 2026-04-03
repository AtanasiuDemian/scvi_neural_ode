from typing import Optional, Union

from lightning.pytorch.accelerators import Accelerator

from scvi_neural_ode.train import Trainer, TrainingPlan


class TrainingMixin:
    def train(
        self,
        n_epochs: Optional[int] = 400,
        lr: Optional[float] = 1e-3,
        accelerator: Union[str, Accelerator] = "auto",
        train_size: float = 0.9,
        validation_size: Optional[float] = None,
        batch_size: int = 128,
        plan_kwargs: Optional[dict] = None,
        data_loader_kwargs: Optional[dict] = None,
        **trainer_kwargs,
    ):
        plan_kwargs = plan_kwargs if isinstance(plan_kwargs, dict) else dict()
        data_loader_kwargs = data_loader_kwargs if isinstance(data_loader_kwargs, dict) else dict()
        data_module = self._data_module_cls(
            self.adata,
            train_size=train_size,
            validation_size=validation_size,
            batch_size=batch_size,
            **data_loader_kwargs,
        )
        data_module.setup()
        if data_module.n_train == self.adata.n_obs:
            # Train on all data
            plan_kwargs.update({"metric_to_monitor": "train_loss"})
        training_plan = TrainingPlan(self.module, lr, **plan_kwargs)
        trainer = Trainer(n_epochs=n_epochs, accelerator=accelerator, **trainer_kwargs)
        trainer.fit(model=training_plan, datamodule=data_module)
        self.train_indices = data_module.train_idx
        self.test_indices = data_module.test_idx
        self.validation_indices = data_module.val_idx
        self.module.eval()
        self.is_trained_ = True
        self.to_device(self.device)
        self.trainer = trainer
