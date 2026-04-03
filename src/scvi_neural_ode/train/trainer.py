from typing import Iterable, Optional, Union

import pytorch_lightning as pl
from lightning.pytorch.accelerators import Accelerator
from pytorch_lightning.loggers.logger import Logger


class Trainer(pl.Trainer):
    def __init__(
        self,
        logger: Union[Logger, Iterable[Logger], bool] = False,
        accelerator: Union[str, Accelerator] = "auto",
        n_epochs: Optional[int] = None,
        log_save_dir: Optional[str] = None,
        benchmark: bool = True,
        **kwargs,
    ):
        super().__init__(
            logger=logger,
            accelerator=accelerator,
            max_epochs=n_epochs,
            default_root_dir=log_save_dir,
            benchmark=benchmark,
            **kwargs,
        )
