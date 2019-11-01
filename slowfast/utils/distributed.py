#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Distributed helpers."""

import torch
import torch.distributed as dist

from slowfast.utils.env import setup_dist_environment

setup_dist_environment()


def all_gather(tensors):
    """
    All gathers the provided tensors from all processes across machines.
    Args:
        tensors (list): tensors to perform all gather across all processes in
        all machines.
    """

    gather_list = []
    output_tensor = []
    world_size = dist.get_world_size()
    for tensor in tensors:
        tensor_placeholder = [
            torch.ones_like(tensor) for _ in range(world_size)
        ]
        dist.all_gather(tensor_placeholder, tensor, async_op=False)
        gather_list.append(tensor_placeholder)
    for gathered_tensor in gather_list:
        output_tensor.append(torch.cat(gathered_tensor, dim=0))
    return output_tensor


def all_reduce(tensors, average=True):
    """
    All reduce the provided tensors from all processes across machines.
    Args:
        tensors (list): tensors to perform all reduce across all processes in
        all machines.
        average (bool): scales the reduced tensor by the number of overall
        processes across all machines.
    """

    for tensor in tensors:
        dist.all_reduce(tensor, async_op=False)
    if average:
        world_size = dist.get_world_size()
        for tensor in tensors:
            tensor.mul_(1.0 / world_size)
    return tensors


def init_process_group(
    local_rank,
    local_world_size,
    shard_id,
    num_shards,
    init_method,
    dist_backend="nccl",
):
    """
    Initializes the default process group.
    Args:
        local_rank (int): the rank on the current local machine.
        local_world_size (int): the world size (number of processes running) on
        the current local machine.
        shard_id (int): the shard index (machine rank) of the current machine.
        num_shards (int): number of shards for distributed training.
        init_method (string): supporting three different methods for
            initializing process groups:
            "file": use shared file system to initialize the groups across
            different processes.
            "tcp": use tcp address to initialize the groups across different
        dist_backend (string): backend to use for distributed training. Options
            includes gloo, mpi and nccl, the details can be found here:
            https://pytorch.org/docs/stable/distributed.html
    """
    # Sets the GPU to use.
    torch.cuda.set_device(local_rank)
    # Initialize the process group.
    proc_rank = local_rank + shard_id * local_world_size
    world_size = local_world_size * num_shards
    dist.init_process_group(
        backend=dist_backend,
        init_method=init_method,
        world_size=world_size,
        rank=proc_rank,
    )


def is_master_proc(num_gpus=8):
    """
    Determines if the current process is the master process.
    """
    if torch.distributed.is_initialized():
        return dist.get_rank() % num_gpus == 0
    else:
        return True
