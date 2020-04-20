#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""BatchNorm (BN) utility functions and custom batch-size BN implementations"""

from functools import partial
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.autograd.function import Function

import slowfast.utils.distributed as du


def get_norm(cfg):
    """
    Args:
        cfg (CfgNode): model building configs, details are in the comments of
            the config file.
    Returns:
        nn.Module: the normalization layer.
    """
    if cfg.BN.NORM_TYPE == "batchnorm":
        return nn.BatchNorm3d
    elif cfg.BN.NORM_TYPE == "sub_batchnorm":
        return partial(SubBatchNorm3d, num_splits=cfg.BN.NUM_SPLITS)
    elif cfg.BN.NORM_TYPE == "sync_batchnorm":
        return partial(
            NaiveSyncBatchNorm3d, num_sync_devices=cfg.BN.NUM_SYNC_DEVICES
        )
    else:
        raise NotImplementedError(
            "Norm type {} is not supported".format(cfg.BN.NORM_TYPE)
        )


class SubBatchNorm3d(nn.Module):
    """
    The standard BN layer computes stats across all examples in a GPU. In some
    cases it is desirable to compute stats across only a subset of examples
    (e.g., in multigrid training https://arxiv.org/abs/1912.00998).
    SubBatchNorm3d splits the batch dimension into N splits, and run BN on
    each of them separately (so that the stats are computed on each subset of
    examples (1/N of batch) independently. During evaluation, it aggregates
    the stats from all splits into one BN.
    """

    def __init__(self, num_splits, **args):
        """
        Args:
            num_splits (int): number of splits.
            args (list): other arguments.
        """
        super(SubBatchNorm3d, self).__init__()
        self.num_splits = num_splits
        num_features = args["num_features"]
        # Keep only one set of weight and bias.
        if args.get("affine", True):
            self.affine = True
            args["affine"] = False
            self.weight = torch.nn.Parameter(torch.ones(num_features))
            self.bias = torch.nn.Parameter(torch.zeros(num_features))
        else:
            self.affine = False
        self.bn = nn.BatchNorm3d(**args)
        args["num_features"] = num_features * num_splits
        self.split_bn = nn.BatchNorm3d(**args)

    def _get_aggregated_mean_std(self, means, stds, n):
        """
        Calculate the aggregated mean and stds.
        Args:
            means (tensor): mean values.
            stds (tensor): standard deviations.
            n (int): number of sets of means and stds.
        """
        mean = means.view(n, -1).sum(0) / n
        std = (
            stds.view(n, -1).sum(0) / n
            + ((means.view(n, -1) - mean) ** 2).view(n, -1).sum(0) / n
        )
        return mean.detach(), std.detach()

    def aggregate_stats(self):
        """
        Synchronize running_mean, and running_var. Call this before eval.
        """
        if self.split_bn.track_running_stats:
            (
                self.bn.running_mean.data,
                self.bn.running_var.data,
            ) = self._get_aggregated_mean_std(
                self.split_bn.running_mean,
                self.split_bn.running_var,
                self.num_splits,
            )

    def forward(self, x):
        if self.training:
            n, c, t, h, w = x.shape
            x = x.view(n // self.num_splits, c * self.num_splits, t, h, w)
            x = self.split_bn(x)
            x = x.view(n, c, t, h, w)
        else:
            x = self.bn(x)
        if self.affine:
            x = x * self.weight.view((-1, 1, 1, 1))
            x = x + self.bias.view((-1, 1, 1, 1))
        return x


class GroupGather(Function):
    """
    GroupGather performs all gather on each of the local process/ GPU groups.
    """

    @staticmethod
    def forward(ctx, input, num_sync_devices, num_groups):
        """
        Perform forwarding, gathering the stats across different process/ GPU
        group.
        """
        ctx.num_sync_devices = num_sync_devices
        ctx.num_groups = num_groups

        input_list = [
            torch.zeros_like(input) for k in range(du.get_local_size())
        ]
        dist.all_gather(
            input_list, input, async_op=False, group=du._LOCAL_PROCESS_GROUP
        )

        inputs = torch.stack(input_list, dim=0)
        if num_groups > 1:
            rank = du.get_local_rank()
            group_idx = rank // num_sync_devices
            inputs = inputs[
                group_idx
                * num_sync_devices : (group_idx + 1)
                * num_sync_devices
            ]
        inputs = torch.sum(inputs, dim=0)
        return inputs

    @staticmethod
    def backward(ctx, grad_output):
        """
        Perform backwarding, gathering the gradients across different process/ GPU
        group.
        """
        grad_output_list = [
            torch.zeros_like(grad_output) for k in range(du.get_local_size())
        ]
        dist.all_gather(
            grad_output_list,
            grad_output,
            async_op=False,
            group=du._LOCAL_PROCESS_GROUP,
        )

        grads = torch.stack(grad_output_list, dim=0)
        if ctx.num_groups > 1:
            rank = du.get_local_rank()
            group_idx = rank // ctx.num_sync_devices
            grads = grads[
                group_idx
                * ctx.num_sync_devices : (group_idx + 1)
                * ctx.num_sync_devices
            ]
        grads = torch.sum(grads, dim=0)
        return grads, None, None


class NaiveSyncBatchNorm3d(nn.BatchNorm3d):
    def __init__(self, num_sync_devices, **args):
        """
        Naive version of Synchronized 3D BatchNorm.
        Args:
            num_sync_devices (int): number of device to sync.
            args (list): other arguments.
        """
        self.num_sync_devices = num_sync_devices
        if self.num_sync_devices > 0:
            assert du.get_local_size() % self.num_sync_devices == 0, (
                du.get_local_size(),
                self.num_sync_devices,
            )
            self.num_groups = du.get_local_size() // self.num_sync_devices
        else:
            self.num_sync_devices = du.get_local_size()
            self.num_groups = 1
        super(NaiveSyncBatchNorm3d, self).__init__(**args)

    def forward(self, input):
        if du.get_local_size() == 1 or not self.training:
            return super().forward(input)

        assert input.shape[0] > 0, "SyncBatchNorm does not support empty inputs"
        C = input.shape[1]
        mean = torch.mean(input, dim=[0, 2, 3, 4])
        meansqr = torch.mean(input * input, dim=[0, 2, 3, 4])

        vec = torch.cat([mean, meansqr], dim=0)
        vec = GroupGather.apply(vec, self.num_sync_devices, self.num_groups) * (
            1.0 / self.num_sync_devices
        )

        mean, meansqr = torch.split(vec, C)
        var = meansqr - mean * mean
        self.running_mean += self.momentum * (mean.detach() - self.running_mean)
        self.running_var += self.momentum * (var.detach() - self.running_var)

        invstd = torch.rsqrt(var + self.eps)
        scale = self.weight * invstd
        bias = self.bias - mean * scale
        scale = scale.reshape(1, -1, 1, 1, 1)
        bias = bias.reshape(1, -1, 1, 1, 1)
        return input * scale + bias
