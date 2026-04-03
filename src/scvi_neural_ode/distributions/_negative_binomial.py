import warnings
from typing import Optional, Tuple, Union

import torch
from torch.distributions import Distribution, Gamma, Poisson, constraints
from torch.distributions.utils import (
    broadcast_all,
    lazy_property,
    logits_to_probs,
    probs_to_logits,
)


def log_nb_positive(x: torch.Tensor, mu: torch.Tensor, theta: torch.Tensor, eps=1e-8):

    if x.ndimension() == 1:
        theta = theta.view(1, theta.size(0))
    log_theta_mu_eps = torch.log(mu + theta + eps)
    output = (
        theta * (torch.log(theta + eps) - log_theta_mu_eps)
        + x * (torch.log(mu + eps) - log_theta_mu_eps)
        + torch.lgamma(x + theta)
        - torch.lgamma(theta)
        - torch.lgamma(x + 1)
    )

    return output


def _convert_counts_logits_to_mean_disp(total_count, logits):
    return torch.exp(-logits) * total_count, total_count


class NB(Distribution):

    arg_constraints = {
        "mu": constraints.greater_than_eq(0),
        "theta": constraints.greater_than_eq(0),
    }
    support = constraints.nonnegative_integer

    def __init__(
        self,
        mu: Optional[torch.Tensor] = None,
        theta: Optional[torch.Tensor] = None,
        total_count: Optional[torch.Tensor] = None,
        probs: Optional[torch.Tensor] = None,
        logits: Optional[torch.Tensor] = None,
        validate_args: bool = False,
    ):
        self._eps = 1e-8
        if logits is not None and probs is not None:
            raise ValueError("Cannot have both p and logits parameters.")

        _has_p = logits is not None or probs is not None
        if (mu is None) == (total_count is None):
            raise ValueError
        using_param_1 = total_count is not None and _has_p
        using_param_2 = mu is not None and theta is None and _has_p
        if using_param_1:
            logits = logits if logits is not None else probs_to_logits(probs, is_binary=True)
            total_count = total_count.type_as(logits)
            # broadcast_all shapes all tensors into the same dimensions.
            total_count, logits = broadcast_all(total_count, logits)
            mu, theta = _convert_counts_logits_to_mean_disp(total_count, logits)
        elif using_param_2:
            logits = (
                logits if logits is not None else probs_to_logits(probs, is_binary=True)
            )  # ln(p) - ln(1-p)
            theta = mu * torch.exp(logits)
            mu, theta = broadcast_all(mu, theta)
        else:
            mu, theta = broadcast_all(mu, theta)

        self.mu = mu
        self.theta = theta
        super().__init__(validate_args=validate_args)

    def log_prob(self, x: torch.Tensor):
        if self._validate_args:
            try:
                self._validate_sample(x)
            except ValueError:
                warnings.warn(
                    "The value argument must be within the support of the distribution",
                    UserWarning,
                )

        return log_nb_positive(x, self.mu, self.theta, self._eps)

    @property
    def mean(self):
        return self.mu

    @property
    def variance(self):
        return self.mean + (self.mean**2) / self.theta

    @lazy_property
    def logits(self):
        return torch.log(self.theta) - torch.log(self.mu)

    @lazy_property
    def probs(self):
        return logits_to_probs(self.logits, is_binary=True)

    @lazy_property
    def _gamma(self):
        # In future get rid of logits for this method, use scale instead.
        return Gamma(concentration=self.theta, rate=torch.exp(self.logits))

    def sample(self, sample_shape: Union[torch.Size, Tuple] = torch.Size()):
        with torch.no_grad():
            rate = self._gamma.sample(sample_shape=sample_shape)
            return Poisson(torch.clamp(rate, max=1e8)).sample()
