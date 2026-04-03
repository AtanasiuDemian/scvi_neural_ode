import torch


def compute_elbo(module, data_loader, **module_kwargs):
    elbo = 0.0
    for tensors in data_loader:
        loss_dict = module.loss(tensors, **module_kwargs)
        reconstruction_error = loss_dict["reconstruction_error"]
        kl_local = loss_dict["kl_local"]
        elbo += torch.sum(reconstruction_error + kl_local).item()
    kl_global = loss_dict["kl_global"]
    n_obs = len(data_loader.indices)
    elbo += kl_global

    return (elbo / n_obs).item()


def compute_reconstruction_error(module, data_loader, return_mean: bool = True, **module_kwargs):
    rec_loss, n_obs = 0.0, 0
    rec_loss_vals = []
    for tensors in data_loader:
        loss_dict = module.loss(tensors, **module_kwargs)
        rec_loss_tensor = loss_dict["reconstruction_error"].detach()
        if return_mean:
            n_obs += len(rec_loss_tensor)
            rec_loss += rec_loss_tensor.sum().item()
        else:
            rec_loss_vals += [rec_loss_tensor]

    return rec_loss / n_obs if return_mean else torch.cat(rec_loss_vals).numpy()
