import torch


def get_step_size(step_size, t1, t2, t_size):
    """
    scTour (Li, 2023) implementation
    """
    if step_size is None:
        return {}
    else:
        step_size = (t2 - t1) / t_size / step_size
        return dict(step_size=step_size)


def unique_index(x):
    """
    Taken from LatentVelo (Farrell et al, 2023), c0c1248b7e.
    Find the index of the unique times for the ODE solver.
    """
    sort_index = torch.argsort(x)

    sorted_x = x[sort_index]
    index = torch.Tensor(
        [torch.max(torch.where(sorted_x == i)[0]) for i in torch.unique(sorted_x)]
    ).long()
    return sort_index, index
