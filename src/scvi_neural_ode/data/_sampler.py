from typing import List, Optional

import numpy as np
import torch


class BatchSampler(torch.utils.data.sampler.Sampler):
    def __init__(
        self,
        indices: np.ndarray,
        batch_size: int,
        shuffle: bool,
        drop_last: bool,
        indices_in_every_batch: Optional[List[int]] = None,
    ):
        self.indices = indices
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

        if indices_in_every_batch is None:
            self.indices_in_every_batch = []
        else:
            # If indices already contains indices_in_every_batch, remove them.
            mask = np.isin(self.indices, indices_in_every_batch)
            self.indices = self.indices[~mask]
            self.indices_in_every_batch = indices_in_every_batch

    def __iter__(self):
        if self.shuffle is True:
            idx = torch.randperm(len(self.indices)).tolist()
        else:
            idx = torch.arange(len(self.indices)).tolist()

        if self.drop_last is True:
            n_without_last = (len(idx) // self.batch_size) * self.batch_size
            idx = idx[:n_without_last]

        data_iter = iter(
            [
                np.concatenate(
                    (
                        self.indices_in_every_batch,
                        self.indices[idx[i : i + self.batch_size]],
                    )
                ).astype(int)
                for i in range(0, len(idx), self.batch_size)
            ]
        )
        return data_iter

    def __len__(self):
        """Return number of batches"""
        if self.drop_last is True:
            return len(self.indices) // self.batch_size
        return len(self.indices) // self.batch_size + 1
