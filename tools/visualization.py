#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import numpy as np
import torch

import slowfast.datasets.utils as data_utils
import slowfast.utils.checkpoint as cu
import slowfast.utils.distributed as du
import slowfast.utils.logging as logging
import slowfast.utils.misc as misc
import slowfast.visualization.tensorboard_vis as tb
from slowfast.datasets import loader
from slowfast.models import build_model

logger = logging.get_logger(__name__)


def run_visualization(vis_loader, model, cfg, writer=None):
    n_devices = cfg.NUM_GPUS * cfg.NUM_SHARDS

    global_idx = 0
    for inputs, _, _, _ in vis_loader:
        # Transfer the data to the current GPU device.
        if isinstance(inputs, (list,)):
            for i in range(len(inputs)):
                inputs[i] = inputs[i].cuda(non_blocking=True)
        else:
            inputs = inputs.cuda(non_blocking=True)

        inputs = du.all_gather_unaligned(inputs)

        if writer is not None:
            total_vids = 0
            for i in range(n_devices):
                cur_input = inputs[i]
                cur_batch_size = cur_input[0].shape[0]

                for cur_batch_idx in range(cur_batch_size):
                    global_idx += 1
                    total_vids += 1
                    for path_idx, input_pathway in enumerate(cur_input):
                        if cfg.TEST.DATASET == "ava" and cfg.AVA.BGR:
                            video = input_pathway[cur_batch_idx, [2, 1, 0], ...]
                        else:
                            video = input_pathway[cur_batch_idx]
                        # Permute to (T, H, W, C) from (C, T, H, W).
                        video = video.permute(1, 2, 3, 0)
                        video = data_utils.revert_tensor_normalize(
                            video.cpu(), cfg.DATA.MEAN, cfg.DATA.STD
                        )
                        video = video.permute(0, 3, 1, 2).unsqueeze(0)
                        writer.add_video(
                            video,
                            tag="Input {}/Input from pathway {}".format(
                                global_idx, path_idx + 1
                            ),
                        )

            logger.info("Visualized {} videos...".format(total_vids))


def visualize(cfg):
    """
    Perform layer weights and activations visualization on the model.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    """
    if cfg.TENSORBOARD.ENABLE and cfg.TENSORBOARD.MODEL_VIS.ENABLE:
        # Set up environment.
        du.init_distributed_training(cfg)
        # Set random seed from configs.
        np.random.seed(cfg.RNG_SEED)
        torch.manual_seed(cfg.RNG_SEED)

        # Setup logging format.
        logging.setup_logging(cfg.OUTPUT_DIR)

        # Print config.
        logger.info("Model Visualization with config:")
        logger.info(cfg)

        # Build the video model and print model statistics.
        model = build_model(cfg)
        if du.is_master_proc() and cfg.LOG_MODEL_INFO:
            misc.log_model_info(model, cfg, is_train=False)

        cu.load_test_checkpoint(cfg, model)

        # Create video testing loaders.
        vis_loader = loader.construct_loader(cfg, "test")
        logger.info(
            "Visualize model for {} data points".format(len(vis_loader))
        )

        if cfg.DETECTION.ENABLE:
            assert cfg.NUM_GPUS == cfg.TEST.BATCH_SIZE

        # Set up writer for logging to Tensorboard format.
        if du.is_master_proc(cfg.NUM_GPUS * cfg.NUM_SHARDS):
            writer = tb.TensorboardWriter(cfg)
        else:
            writer = None

        # Run visualization on the model
        run_visualization(vis_loader, model, cfg, writer)

        if writer is not None:
            writer.close()
