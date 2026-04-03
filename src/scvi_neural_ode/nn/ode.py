import torch
import torch.nn as nn

from .base import FCLayers


class ODEfunc(nn.Module):
    """
    This network needs to be an nn Module to be used by the odeint solver (see scTour implementation)
    """

    def __init__(self, n_input, n_hidden, n_output, n_batch=0, **layer_kwargs):
        super().__init__()
        self.fc1 = FCLayers(
            n_in=n_input,
            n_hidden=n_hidden,
            n_out=n_hidden,
            dropout_rate=0,
            use_batch_norm=False,
            n_cat_list=[n_batch],
            **layer_kwargs,
        )
        self.fc2 = nn.Linear(n_hidden, n_output)

    def forward(self, t: torch.Tensor, x: torch.Tensor, *batch):
        return self.fc2(self.fc1(x, *batch))


class TimeEncoder(nn.Module):
    def __init__(self, n_input, n_hidden=128, n_layers=1, n_cat_list=None):
        super().__init__()
        self.fc = FCLayers(
            n_in=n_input,
            n_hidden=n_hidden,
            n_out=n_hidden,
            dropout_rate=0,
            use_batch_norm=False,
            n_layers=n_layers,
            use_layer_norm=True,
            n_cat_list=n_cat_list,
        )
        self.layers = nn.Sequential(
            nn.Linear(n_hidden, 1),
            nn.Sigmoid(),
        )

    def forward(self, x, batch_index=None, cont_covs=None, cat_covs=None):
        if cont_covs is not None:
            x = torch.cat([x, cont_covs], dim=-1)
        if cat_covs is not None:
            cat_input = torch.split(cat_covs, 1, dim=1)
        else:
            cat_input = tuple()

        return self.layers(self.fc(x, batch_index, *cat_input))
