# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Functions for benchmarks.
"""

import numpy as np
import pprint
import torch
import tqdm
from fvcore.common.timer import Timer

import slowfast.utils.logging as logging
import slowfast.utils.misc as misc
from slowfast.datasets import loader
from slowfast.utils.env import setup_environment

logger = logging.get_logger(__name__)


def benchmark_data_loading(cfg):
    """
    Benchmark the speed of data loading in PySlowFast.
    Args:

        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    """
    # Set up environment.
    setup_environment()
    # Set random seed from configs.
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)

    # Setup logging format.
    logging.setup_logging(cfg.OUTPUT_DIR)

    # Print config.
    logger.info("Benchmark data loading with config:")
    logger.info(pprint.pformat(cfg))

    timer = Timer()
    dataloader = loader.construct_loader(cfg, "train")
    logger.info(
        "Initialize loader using {:.2f} seconds.".format(timer.seconds())
    )
    # Total batch size across different machines.
    batch_size = cfg.TRAIN.BATCH_SIZE * cfg.NUM_SHARDS
    log_period = cfg.BENCHMARK.LOG_PERIOD
    epoch_times = []
    # Test for a few epochs.
    for cur_epoch in range(cfg.BENCHMARK.NUM_EPOCHS):
        timer = Timer()
        timer_epoch = Timer()
        iter_times = []
        if cfg.BENCHMARK.SHUFFLE:
            loader.shuffle_dataset(dataloader, cur_epoch)
        for cur_iter, _ in enumerate(tqdm.tqdm(dataloader)):
            if cur_iter > 0 and cur_iter % log_period == 0:
                iter_times.append(timer.seconds())
                ram_usage, ram_total = misc.cpu_mem_usage()
                logger.info(
                    "Epoch {}: {} iters ({} videos) in {:.2f} seconds. "
                    "RAM Usage: {:.2f}/{:.2f} GB.".format(
                        cur_epoch,
                        log_period,
                        log_period * batch_size,
                        iter_times[-1],
                        ram_usage,
                        ram_total,
                    )
                )
                timer.reset()
        epoch_times.append(timer_epoch.seconds())
        ram_usage, ram_total = misc.cpu_mem_usage()
        logger.info(
            "Epoch {}: in total {} iters ({} videos) in {:.2f} seconds. "
            "RAM Usage: {:.2f}/{:.2f} GB.".format(
                cur_epoch,
                len(dataloader),
                len(dataloader) * batch_size,
                epoch_times[-1],
                ram_usage,
                ram_total,
            )
        )
        logger.info(
            "Epoch {}: on average every {} iters ({} videos) take {:.2f}/{:.2f} "
            "(avg/std) seconds.".format(
                cur_epoch,
                log_period,
                log_period * batch_size,
                np.mean(iter_times),
                np.std(iter_times),
            )
        )
    logger.info(
        "On average every epoch ({} videos) takes {:.2f}/{:.2f} "
        "(avg/std) seconds.".format(
            len(dataloader) * batch_size,
            np.mean(epoch_times),
            np.std(epoch_times),
        )
    )
