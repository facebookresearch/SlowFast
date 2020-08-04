#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Multiprocessing helpers."""

import torch
import os

def run(
    local_rank, num_proc, func, init_method, shard_id, num_shards, backend, cfg
):
    """
    Runs a function from a child process.
    Args:
        local_rank (int): rank of the current process on the current machine.
        num_proc (int): number of processes per machine.
        func (function): function to execute on each of the process.
        init_method (string): method to initialize the distributed training.
            TCP initialization: equiring a network address reachable from all
            processes followed by the port.
            Shared file-system initialization: makes use of a file system that
            is shared and visible from all machines. The URL should start with
            file:// and contain a path to a non-existent file on a shared file
            system.
        shard_id (int): the rank of the current machine.
        num_shards (int): number of overall machines for the distributed
            training job.
        backend (string): three distributed backends ('nccl', 'gloo', 'mpi') are
            supports, each with different capabilities. Details can be found
            here:
            https://pytorch.org/docs/stable/distributed.html
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    """
    # Initialize the process group.
    world_size = num_proc * num_shards
    rank = shard_id * num_proc + local_rank
    print("RUN local_rank {} rank {} world_size {} shard_id {} num_shards {} init_method {} dist_backed {}".format(
        local_rank, rank, world_size, shard_id, num_shards, init_method, backend
    ))

    try:
        master_addr = os.environ.get("MASTER_ADDR", None)
        master_port = os.environ.get("MASTER_PORT", None)
        print("In run() MASTER {} PORT {}".format(master_addr, master_port))

        torch.distributed.init_process_group(
            backend=backend,
            init_method=init_method,
            world_size=world_size,
            rank=rank,
        )
    except Exception as e:
        raise e

    torch.cuda.set_device(local_rank)
    print("Call func - shard {} ".format(shard_id))
    func(cfg)
