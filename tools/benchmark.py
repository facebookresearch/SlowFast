#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
A script to benchmark data loading.
"""

import numpy as np
import pprint
import psutil
import torch
import tqdm
from fvcore.common.timer import Timer

import slowfast.utils.logging as logging
import slowfast.utils.multiprocessing as mpu
from slowfast.datasets import loader
from slowfast.utils.env import setup_environment
from slowfast.utils.parser import load_config, parse_args

logger = logging.get_logger(__name__)


def benchmark_data(cfg):
    # Set up environment.
    setup_environment()
    # Set random seed from configs.
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)

    # Setup logging format.
    logging.setup_logging()

    # Print config.
    logger.info("Benchmark data loading with config:")
    logger.info(pprint.pformat(cfg))

    timer = Timer()
    dataloader = loader.construct_loader(cfg, "train")
    logger.info(
        "Initialize loader using {:.2f} seconds.".format(timer.seconds())
    )
    batch_size = cfg.TRAIN.BATCH_SIZE
    log_period = cfg.BENCHMARK.LOG_PERIOD
    epoch_times = []
    # Test for a few epochs.
    for cur_epoch in range(cfg.BENCHMARK.NUM_EPOCHS):
        timer = Timer()
        timer_epoch = Timer()
        iter_times = []
        for cur_iter, _ in enumerate(tqdm.tqdm(dataloader)):
            if cur_iter > 0 and cur_iter % log_period == 0:
                iter_times.append(timer.seconds())
                vram = psutil.virtual_memory()
                logger.info(
                    "Epoch {}: {} iters ({} videos) in {:.2f} seconds. "
                    "RAM Usage: {:.2f}/{:.2f} GB.".format(
                        cur_epoch,
                        log_period,
                        log_period * batch_size,
                        iter_times[-1],
                        (vram.total - vram.available) / 1024 ** 3,
                        vram.total / 1024 ** 3,
                    )
                )
                timer.reset()
        epoch_times.append(timer_epoch.seconds())
        vram = psutil.virtual_memory()
        logger.info(
            "Epoch {}: in total {} iters ({} videos) in {:.2f} seconds. "
            "RAM Usage: {:.2f}/{:.2f} GB.".format(
                cur_epoch,
                len(dataloader),
                len(dataloader) * batch_size,
                epoch_times[-1],
                (vram.total - vram.available) / 1024 ** 3,
                vram.total / 1024 ** 3,
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


def main():
    args = parse_args()
    cfg = load_config(args)

    if cfg.NUM_GPUS > 1:
        torch.multiprocessing.spawn(
            mpu.run,
            nprocs=cfg.NUM_GPUS,
            args=(
                cfg.NUM_GPUS,
                benchmark_data,
                args.init_method,
                cfg.SHARD_ID,
                cfg.NUM_SHARDS,
                cfg.DIST_BACKEND,
                cfg,
            ),
            daemon=False,
        )
    else:
        benchmark_data(cfg=cfg)


if __name__ == "__main__":
    main()
