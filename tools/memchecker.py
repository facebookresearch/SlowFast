#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Multi-view test a video classification model."""

import numpy as np
import os
import torch

import slowfast.utils.distributed as du
import slowfast.utils.logging as logging
import slowfast.utils.misc as misc
from slowfast.datasets import loader

logger = logging.get_logger(__name__)
import time as TT


@torch.no_grad()
def perform_check(mcheck_loader, cfg):
    """
    For classification:
    Perform mutli-view testing that uniformly samples N clips from a video along
    its temporal axis. For each clip, it takes 3 crops to cover the spatial
    dimension, followed by averaging the softmax scores across all Nx3 views to
    form a video-level prediction. All video predictions are compared to
    ground-truth labels and the final testing performance is logged.
    For detection:
    Perform fully-convolutional testing on the full frames without crop.
    Args:
        test_loader (loader): video testing loader.
        model (model): the pretrained video model to test.
        test_meter (TestMeter): testing meters to log and ensemble the testing
            results.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    """
    # DALI case
    if cfg.DALI_ENABLE:
        print("DALI iteration is running -> EXIT")
        return
    start_T = TT.time()
    for cur_iter, (inputs, labels, index, time, meta) in enumerate(mcheck_loader):
        end_T = TT.time()
        # print(f"iter{cur_iter}, time differene: {end_T-start_T}")
        start_T = end_T
        # Transfer the data to the current GPU device.
        # continue
        if cfg.NUM_GPUS:
            if isinstance(inputs, (list,)):
                for i in range(len(inputs)):
                    if isinstance(inputs[i], (list,)):
                        for j in range(len(inputs[i])):
                            inputs[i][j] = inputs[i][j].cuda(non_blocking=True)
                    else:
                        inputs[i] = inputs[i].cuda(non_blocking=True)
            else:
                inputs = inputs.cuda(non_blocking=True)
            if not isinstance(labels, list):
                labels = labels.cuda(non_blocking=True)
                index = index.cuda(non_blocking=True)
                time = time.cuda(non_blocking=True)
            for key, val in meta.items():
                if isinstance(val, (list,)):
                    for i in range(len(val)):
                        val[i] = val[i].cuda(non_blocking=True)
                else:
                    meta[key] = val.cuda(non_blocking=True)
        # print(len(inputs))
        # print(inputs[0][0].shape)
        # print(labels.shape)
        # print(index.shape)
        # print(time.shape)
    # print("one epoch done")
    return "done"


def memcheck(cfg):
    """
    Perform multi-view testing on the pretrained video model.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    """
    # Set up environment.
    du.init_distributed_training(cfg.NUM_GPUS, cfg.SHARD_ID)
    # Set random seed from configs.
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)

    # Setup logging format.
    logging.setup_logging(cfg.OUTPUT_DIR)

    # Print config.
    logger.info("MemCheck")
    # logger.info("MemCheck with config:")
    # logger.info(cfg)

    # Create video testing loaders.
    # logger.info(f"Dataloaer Split will be changed from 'memcheck' to 'train'.")
    memcheck_loader = loader.construct_loader(cfg, "memcheck")
    logger.info("Memory check model for {} iterations".format(len(memcheck_loader)))
    # # Perform memory checkiong on the entire dataset.

    start_epoch = 0
    start_epoch_time = TT.time()
    for cur_epoch in range(start_epoch, cfg.SOLVER.MAX_EPOCH):
        performed_return = perform_check(memcheck_loader, cfg)
        end_epoch_time = TT.time()
        print(f"{cur_epoch}, {end_epoch_time-start_epoch_time}")
        start_epoch_time = TT.time()

    # put the gpu_mem_usage() return in dataloader
    # misc.gpu_mem_usage(),
    return f"memcheck is ended-> {performed_return}"