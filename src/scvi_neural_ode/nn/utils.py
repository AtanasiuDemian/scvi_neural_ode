import torch


def one_hot(index, n_cat):  # helper function from scvi
    onehot = torch.zeros(index.size(0), n_cat, device=index.device)
    onehot.scatter_(1, index.type(torch.long), 1)
    # .scatter_ writes all values from the tensor 1 (of ones) into onehot at the indices specified in the index tensor
    # Don't understand what it does when dim is 1 (first argument)
    return onehot.type(torch.float32)
