from abc import abstractmethod

import torch.nn as nn


class BaseModule(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()

    @abstractmethod
    def generative(self, *args, **kwargs):
        pass

    @abstractmethod
    def inference(self, *args, **kwargs):
        pass

    @abstractmethod
    def forward(self, tensors, **kwargs):
        pass

    @abstractmethod
    def loss(self, tensors, **kwargs):
        pass

    @property
    def device(self):
        device = list(set(p.device for p in self.parameters()))
        if len(device) > 1:
            raise RuntimeError("Tensors on distinct devices.")

        return device[0]
