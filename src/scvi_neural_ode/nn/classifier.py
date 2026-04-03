import torch.nn as nn

from .base import FCLayers


class Classifier(nn.Module):
    def __init__(
        self,
        n_input: int,
        n_labels: int,
        n_hidden: int = 128,
        n_layers: int = 1,
        dropout_rate: float = 0.1,
        use_logits: bool = False,
        use_batch_norm: bool = True,
        use_layer_norm: bool = False,
        activation_fn: nn.Module = nn.ReLU,
        **kwargs,
    ):
        super().__init__()
        self.use_logits = use_logits
        layers = []

        if n_hidden > 0 and n_layers > 0:
            layers.append(
                FCLayers(
                    n_in=n_input,
                    n_out=n_hidden,
                    n_layers=n_layers,
                    n_hidden=n_hidden,
                    dropout_rate=dropout_rate,
                    use_batch_norm=use_batch_norm,
                    use_layer_norm=use_layer_norm,
                    activation_fn=activation_fn,
                    **kwargs,
                )
            )
        else:
            n_hidden = n_input

        layers.append(nn.Linear(n_hidden, n_labels))

        if not use_logits:
            layers.append(nn.Softmax(dim=-1))

        self.classifier = nn.Sequential(*layers)

    def forward(self, x):
        return self.classifier(x)
