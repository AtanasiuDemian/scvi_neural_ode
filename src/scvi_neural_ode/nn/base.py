from collections import OrderedDict
from typing import Iterable

import torch
from torch import nn as nn
from torch.distributions import Normal

from .utils import one_hot


class FCLayers(nn.Module):
    """
    Function to build a fully connected neural network.
    n_cat_list : List containing the number of categories for each category of interest.
    """

    def __init__(
        self,
        n_in: int,
        n_out: int,
        n_cat_list: Iterable[int] = None,
        n_layers: int = 1,
        n_hidden: int = 128,
        use_activation: bool = True,
        use_batch_norm: bool = True,
        use_layer_norm: bool = False,
        dropout_rate: float = 0.1,
        bias: bool = True,
        activation_fn: nn.Module = nn.ReLU,
    ):
        # For now assume it injects covariates into each layer.
        super().__init__()
        layers_dim = [n_in] + (n_layers - 1) * [n_hidden] + [n_out]

        if n_cat_list is not None:
            self.n_cat_list = [n_cat if n_cat > 1 else 0 for n_cat in n_cat_list]
        else:
            self.n_cat_list = []

        cat_dim = sum(self.n_cat_list)
        self.fc_layers = nn.Sequential(
            OrderedDict(
                [
                    (
                        "Layer_{}".format(i),
                        nn.Sequential(
                            nn.Linear(
                                n_in + cat_dim,
                                n_out,
                                bias=bias,
                            ),
                            (
                                nn.BatchNorm1d(n_out, momentum=0.01, eps=0.001)
                                if use_batch_norm
                                else None
                            ),
                            (
                                nn.LayerNorm(n_out, elementwise_affine=False)
                                if use_layer_norm
                                else None
                            ),
                            activation_fn() if use_activation else None,
                            nn.Dropout(p=dropout_rate) if dropout_rate > 0 else None,
                        ),
                    )
                    for i, (n_in, n_out) in enumerate(zip(layers_dim[:-1], layers_dim[1:]))
                ]
            )
        )

    def new_input_nodes_hook(self, n_nodes: int):
        self.hooks = []

        def _hook_new_node(grad):
            new_grad = torch.zeros_like(grad)
            new_grad[:, -n_nodes:] = grad[:, -n_nodes:]
            return new_grad

        w = self.fc_layers[0][0].weight.register_hook(_hook_new_node)
        self.hooks.append(w)

    def set_online_update_hooks(self, hook_first_layer=True):
        self.hooks = []

        def _hook_fn_weight(grad):
            categorical_dims = sum(self.n_cat_list)
            new_grad = torch.zeros_like(grad)
            if categorical_dims > 0:
                new_grad[:, -categorical_dims + 1 :] = grad[:, -categorical_dims + 1 :]
            return new_grad

        def _hook_fn_zero_out(grad):
            return grad * 0

        for i, layers in enumerate(self.fc_layers):
            for layer in layers:
                if i == 0 and not hook_first_layer:
                    continue
                if isinstance(layer, nn.Linear):
                    # if self.inject_into_layer(i):
                    w = layer.weight.register_hook(_hook_fn_weight)
                    self.hooks.append(w)
                    b = layer.bias.register_hook(_hook_fn_zero_out)
                    self.hooks.append(b)

    def forward(self, x: torch.Tensor, *cat_list: int):

        one_hot_cat_list = []
        for n_cat, cat in zip(self.n_cat_list, cat_list):
            if n_cat and cat is None:
                raise ValueError("cat not provided while n_cat != 0 in init. params.")
            if n_cat > 1:
                if cat.size(1) != n_cat:
                    if not (cat == cat.long()).all().item():
                        raise ValueError("Categorical input contains non-discrete values.")
                    one_hot_cat = one_hot(cat, n_cat)
                else:
                    one_hot_cat = cat  # cat has already been one_hot encoded.
                one_hot_cat_list += [one_hot_cat]

        for i, layers in enumerate(self.fc_layers):
            for layer in layers:
                if layer is not None:
                    if isinstance(layer, nn.BatchNorm1d):
                        if x.dim() == 3:
                            x = torch.cat([(layer(slice_x)).unsqueeze(0) for slice_x in x], dim=0)
                        else:
                            x = layer(x)
                    else:
                        if isinstance(layer, nn.Linear):
                            if x.dim() == 3:
                                one_hot_cat_list_layer = [
                                    o.unsqueeze(0).expand((x.size(0), o.size(0), o.size(1)))
                                    for o in one_hot_cat_list
                                ]
                            else:
                                one_hot_cat_list_layer = one_hot_cat_list
                            x = torch.cat((x, *one_hot_cat_list_layer), dim=-1)
                        x = layer(x)

        return x


class Encoder(nn.Module):
    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_layers: int = 1,
        n_hidden: int = 128,
        n_cat_list: Iterable[int] = None,
        use_batch_norm: bool = True,
        use_layer_norm: bool = False,
        dropout_rate: float = 0.1,
        **kwargs,
    ):
        super().__init__()
        self.encoder = FCLayers(
            n_in=n_input,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm,
            dropout_rate=dropout_rate,
            **kwargs,
        )
        self.mean_encoder = nn.Linear(n_hidden, n_output)
        self.var_encoder = nn.Linear(n_hidden, n_output)

    def forward(self, x: torch.Tensor, *cat_list: int):
        q_h = self.encoder(x, *cat_list)
        z_mean = self.mean_encoder(q_h)
        # z_var = torch.clamp(torch.exp(self.var_encoder(q_h)), min=1e-4, max=np.exp(4))
        z_var = torch.exp(self.var_encoder(q_h)) + 1e-4
        # z_var = torch.ones_like(z_mean)
        z_sample = Normal(z_mean, z_var.sqrt()).rsample()
        return z_mean, z_var, z_sample


class Decoder(nn.Module):
    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_layers: int = 1,
        n_hidden: int = 128,
        n_cat_list: Iterable[int] = None,
        use_batch_norm: bool = False,
        use_layer_norm: bool = False,
        dropout_rate: float = 0,
        **kwargs,
    ):
        # Note that for the decoder they set the default dropout rate to 0.
        super().__init__()
        self.decoder = FCLayers(
            n_in=n_input,
            n_out=n_hidden,
            n_layers=n_layers,
            n_cat_list=n_cat_list,
            n_hidden=n_hidden,
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm,
            dropout_rate=dropout_rate,
            **kwargs,
        )

        self.prop_decoder = nn.Sequential(nn.Linear(n_hidden, n_output), nn.Softmax(dim=-1))
        # Use neural network only for gene-cell dispersion. Otherwise, infer via gradient methods.

    def forward(self, z: torch.Tensor, *cat_list: int):
        x_h = self.decoder(z, *cat_list)
        x_prop = self.prop_decoder(x_h)

        return x_prop


class LinearDecoder(nn.Module):
    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_cat_list: Iterable[int] = None,
        use_batch_norm: bool = False,
        use_layer_norm: bool = False,
        dropout_rate: float = 0,
        bias: bool = False,
    ):
        super(LinearDecoder, self).__init__()
        self.factor_regressor = FCLayers(
            n_in=n_input,
            n_out=n_output,
            n_cat_list=n_cat_list,
            n_layers=1,
            use_activation=False,
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm,
            bias=bias,
            dropout_rate=dropout_rate,
        )

    def forward(self, z: torch.Tensor, *cat_list: int):
        px = self.factor_regressor(z, *cat_list)
        px_props = torch.softmax(px, dim=-1)

        return px_props
