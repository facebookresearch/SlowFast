#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Helper functions for multigrid training."""

import numpy as np
from torch._six import int_classes as _int_classes
from torch.utils.data.sampler import Sampler


class ShortCycleBatchSampler(Sampler):
    """
    Extend Sampler to support "short cycle" sampling.
    See paper "A Multigrid Method for Efficiently Training Video Models",
    Wu et al., 2019 (https://arxiv.org/abs/1912.00998) for details.
    """

    def __init__(self, sampler, batch_size, drop_last, cfg):
        if not isinstance(sampler, Sampler):
            raise ValueError(
                "sampler should be an instance of "
                "torch.utils.data.Sampler, but got sampler={}".format(sampler)
            )
        if (
            not isinstance(batch_size, _int_classes)
            or isinstance(batch_size, bool)
            or batch_size <= 0
        ):
            raise ValueError(
                "batch_size should be a positive integer value, "
                "but got batch_size={}".format(batch_size)
            )
        if not isinstance(drop_last, bool):
            raise ValueError(
                "drop_last should be a boolean value, but got "
                "drop_last={}".format(drop_last)
            )
        self.sampler = sampler
        self.drop_last = drop_last

        bs_factor = [
            int(
                round(
                    (
                        float(cfg.DATA.TRAIN_CROP_SIZE)
                        / (s * cfg.MULTIGRID.DEFAULT_S)
                    )
                    ** 2
                )
            )
            for s in cfg.MULTIGRID.SHORT_CYCLE_FACTORS
        ]

        self.batch_sizes = [
            batch_size * bs_factor[0],
            batch_size * bs_factor[1],
            batch_size,
        ]

    def __iter__(self):
        counter = 0
        batch_size = self.batch_sizes[0]
        batch = []
        for idx in self.sampler:
            batch.append((idx, counter % 3))
            if len(batch) == batch_size:
                yield batch
                counter += 1
                batch_size = self.batch_sizes[counter % 3]
                batch = []
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        avg_batch_size = sum(self.batch_sizes) / 3.0
        if self.drop_last:
            return int(np.floor(len(self.sampler) / avg_batch_size))
        else:
            return int(np.ceil(len(self.sampler) / avg_batch_size))
