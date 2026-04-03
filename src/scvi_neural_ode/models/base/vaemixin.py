import logging
from typing import Optional, Sequence

import numpy as np
import torch
from anndata import AnnData

from scvi_neural_ode import _CONSTANTS

from .log_likelihood import compute_elbo, compute_reconstruction_error

logger = logging.getLogger(__name__)


class VAEMixin:
    @torch.no_grad()
    def get_elbo(
        self,
        adata: Optional[AnnData] = None,
        indices: Optional[Sequence[int]] = None,
        batch_size: Optional[int] = None,
    ):
        dl = self._make_data_loader(
            adata=adata, indices=indices, batch_size=batch_size, shuffle=False
        )
        elbo = compute_elbo(self.module, dl)

        return -elbo

    @torch.no_grad()
    def get_reconstruction_error(
        self,
        adata: Optional[AnnData] = None,
        indices: Optional[Sequence[int]] = None,
        batch_size: Optional[int] = None,
        return_mean: bool = True,
        **loss_kwargs,
    ):
        dl = self._make_data_loader(
            adata=adata, indices=indices, batch_size=batch_size, shuffle=False
        )
        reconstruction_error = compute_reconstruction_error(
            module=self.module, data_loader=dl, return_mean=return_mean, **loss_kwargs
        )

        return reconstruction_error

    @torch.no_grad()
    def get_latent_representation(
        self,
        adata: Optional[AnnData] = None,
        indices: Optional[Sequence[int]] = None,
        use_mean: bool = True,
        batch_size: Optional[int] = None,
    ):
        """
        "Return latent representation of each cell", i.e. z_{n}.
        """
        if self.is_trained_ is False:
            raise RuntimeError("Need to train model.")

        scdl = self._make_data_loader(
            adata=adata, indices=indices, batch_size=batch_size, shuffle=False
        )
        latent = []
        for tensors in scdl:
            module_outputs = self.module.forward(tensors)
            if use_mean is True:
                z = module_outputs["z_mean"]
            else:
                z = module_outputs["z_sample"]
            latent += [z.cpu()]

        return torch.cat(latent).numpy()

    @torch.no_grad()
    def forward_pass(
        self,
        adata: Optional[AnnData] = None,
        indices: Optional[Sequence[int]] = None,
        **decoder_covariates,
    ):
        """
        Return encoder and decoder output of forward pass through the VAE model.

        Input
        -----
        adata
            AnnData object
        indices
            Indices of samples in adata
        decoder_covariates
            If provided, the latents will be passed through the conditional decoder with these covariates.
            Not all keys need to be provided - the missing ones will be replaced by the default metadata values.
            The result is reflected only in the gene proportions `x_rate`.

        Output
        ------
        dict[str, np.ndarray]

        Example
        -------
        >> model.forward_pass(indices=[1, 2, 3], batch = 2)

        will return a dictionary of values for cells indexed at 1,2,3.
        Gene expression `x_rate` will now contain the predictions using covariate `batch` = 2.
        """
        if not self.is_trained_:
            raise RuntimeError("Need to train model.")
        if adata is None:
            adata = self.adata

        if indices is not None:
            batch_size = len(indices)
        else:
            batch_size = adata.n_obs

        # Need to change this to minibatch operations
        scdl = self._make_data_loader(adata, batch_size=batch_size, indices=indices, shuffle=False)
        tensors = [tensors for tensors in scdl][0]
        outputs = self.module(tensors)
        for key in outputs.keys():
            outputs.update({key: outputs[key].cpu().numpy()})

        if decoder_covariates:
            logger.warning(
                "Decoding with custom decoder covariates, see gene proportions `x_rate`."
            )
            logger_msg = "Decoding with the following covariate values: \n"
            data_registry = adata.uns["data_registry"]
            all_input_keys = []
            for key in data_registry.keys():
                if key != "X":
                    attr_key = data_registry[key][1]
                    if isinstance(attr_key, str):
                        attr_key = [attr_key]
                    all_input_keys += attr_key
            for input_key in all_input_keys:
                if input_key not in decoder_covariates.keys():
                    # If covariate not provided, use the metadata.
                    # NOTE: this is a dodgy formulation, because `input_key` might not come from .obs
                    vals = adata.obs_vector(input_key)
                    decoder_covariates[input_key] = vals if indices is None else vals[indices]
                    logger_msg += f"{input_key}: default. \n"
                else:
                    if isinstance(decoder_covariates[input_key], (list, np.ndarray, torch.Tensor)):
                        logger_msg += f"{input_key}: custom array/tensor. \n"
                    else:
                        logger_msg += f"{input_key}: {decoder_covariates[input_key]} \n"

            z_sample = outputs["z_sample"]
            x_rate = self.decode_with_covariates(z_sample, **decoder_covariates)
            outputs["x_rate"] = x_rate
            logger.info(logger_msg)

        return outputs

    @torch.no_grad()
    def decode_with_covariates(self, z, **cov_dict):
        """
        Decode input `z` of latent variables with given covariates.

        Example
        --------
        If data registry is {BATCH_KEY: ['obs', 'A'], CAT_COV_KEY: ['obs', ['B', 'C']]}
        then call function as
        >> model.decode_with_covariates(z, A=[...], B=[...], C=[...])
        where A,B,C can be either a constant value or a given array.

        This function requires ALL covariates used in the networks.

        Input
        ------
        cov_dict
            Keyword arguments of all covariates used in decoder input.
            (Note: this function needs all covariates because it does not take metadata input.)

        Returns
        -------
        np.ndarray, shape (len(z), n_genes)
            Gene proportions
        """

        def broadcast_to_tensors(x):
            if isinstance(x, (np.integer, int, float)):
                # decode everything with one fixed covariate value.
                y = x * torch.ones((z.shape[0], 1))
            elif isinstance(x, (list, np.ndarray, torch.Tensor)):
                if not isinstance(x, torch.Tensor):
                    y = torch.tensor(x)
                else:
                    y = x
                y = y.float()

                if y.ndim == 1:
                    y = y.unsqueeze(1)
            else:
                raise TypeError("Covariate is of invalid type.")

            return y

        keys = [_CONSTANTS.BATCH_KEY, _CONSTANTS.CAT_COV_KEY, _CONSTANTS.CONT_COV_KEY]
        input_keys = [
            "batch_index",
            "cat_covs",
            "cont_covs",
        ]  # Keyword arguments of `module.generative`
        registry_dict = self.adata.uns["data_registry"]
        no_key_msg = lambda k, s: f"Covariate `{s}` of key `{k}` not found in the input. "

        # Input can be: a constant, custom list or default values.
        # Problem is that it relays to adata while input is z.
        # So require input to contain all keys in data_registry

        input_dict = {}
        for key, input_key in zip(keys, input_keys):
            if key not in registry_dict.keys():
                continue
            _, attr_keys = registry_dict[key]
            # `attr_keys` is either a string or a list.
            if isinstance(attr_keys, str):
                if attr_keys not in cov_dict.keys():
                    raise ValueError(no_key_msg(key, attr_keys))
                attr_val = broadcast_to_tensors(cov_dict[attr_keys])
                del cov_dict[attr_keys]
            elif isinstance(attr_keys, list):
                for k_ in attr_keys:
                    if k_ not in cov_dict.keys():
                        raise ValueError(no_key_msg(key, k_))
                attr_val = torch.cat(
                    [broadcast_to_tensors(cov_dict[k_]) for k_ in attr_keys], dim=1
                )
                for k_ in attr_keys:
                    del cov_dict[k_]
            else:
                raise ValueError(f"Format {attr_keys} of input keys for `{key}` is invalid.")

            input_dict[input_key] = attr_val

        if cov_dict.keys():
            logger.warning(
                f"Keys {list(cov_dict.keys())} are not present in data registry and will be ignored."
            )

        if isinstance(z, np.ndarray):
            z = torch.from_numpy(z)

        generative_outputs = self.module.generative(z, **input_dict)
        x_rate = generative_outputs["x_rate"].cpu().numpy()

        return x_rate
