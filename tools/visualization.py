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
from slowfast.visualization.utils import (
    GetWeightAndActivation,
    process_layer_index_data,
)
from slowfast.visualization.video_visualizer import VideoVisualizer

logger = logging.get_logger(__name__)


def run_visualization(vis_loader, model, cfg, writer=None):
    """
    Run model visualization (weights, activations and model inputs) and visualize
    them on Tensorboard.
    Args:
        vis_loader (loader): video visualization loader.
        model (model): the video model to visualize.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        writer (TensorboardWriter, optional): TensorboardWriter object
            to writer Tensorboard log.
    """
    n_devices = cfg.NUM_GPUS * cfg.NUM_SHARDS
    prefix = "module/" if n_devices > 1 else ""
    # Get a list of selected layer names and indexing.
    layer_ls, indexing_dict = process_layer_index_data(
        cfg.TENSORBOARD.MODEL_VIS.LAYER_LIST, layer_name_prefix=prefix
    )
    logger.info("Start Model Visualization.")
    # Register hooks for activations.
    model_vis = GetWeightAndActivation(model, layer_ls)

    if writer is not None and cfg.TENSORBOARD.MODEL_VIS.MODEL_WEIGHTS:
        layer_weights = model_vis.get_weights()
        writer.plot_weights_and_activations(
            layer_weights, tag="Layer Weights/", heat_map=False
        )

    video_vis = VideoVisualizer(
        cfg.MODEL.NUM_CLASSES,
        cfg.TENSORBOARD.CLASS_NAMES_PATH,
        cfg.TENSORBOARD.MODEL_VIS.TOPK_PREDS,
        cfg.TENSORBOARD.MODEL_VIS.COLORMAP,
    )
    logger.info("Finish drawing weights.")
    global_idx = -1
    for inputs, _, _, meta in vis_loader:
        # Transfer the data to the current GPU device.
        if isinstance(inputs, (list,)):
            for i in range(len(inputs)):
                inputs[i] = inputs[i].cuda(non_blocking=True)
        else:
            inputs = inputs.cuda(non_blocking=True)
        for key, val in meta.items():
            if isinstance(val, (list,)):
                for i in range(len(val)):
                    val[i] = val[i].cuda(non_blocking=True)
            else:
                meta[key] = val.cuda(non_blocking=True)

        if cfg.DETECTION.ENABLE:
            activations, preds = model_vis.get_activations(
                inputs, meta["boxes"]
            )
        else:
            activations, preds = model_vis.get_activations(inputs)

        inputs = du.all_gather_unaligned(inputs)
        activations = du.all_gather_unaligned(activations)
        preds = du.all_gather_unaligned(preds)
        boxes = [None] * n_devices
        if cfg.DETECTION.ENABLE:
            boxes = du.all_gather_unaligned(meta["boxes"])

        if writer is not None:
            total_vids = 0
            for i in range(n_devices):
                cur_input = inputs[i]
                cur_activations = activations[i]
                cur_batch_size = cur_input[0].shape[0]
                cur_preds = preds[i].cpu()
                cur_boxes = boxes[i]
                for cur_batch_idx in range(cur_batch_size):
                    global_idx += 1
                    total_vids += 1
                    if cfg.TENSORBOARD.MODEL_VIS.INPUT_VIDEO:
                        for path_idx, input_pathway in enumerate(cur_input):
                            if cfg.TEST.DATASET == "ava" and cfg.AVA.BGR:
                                video = input_pathway[
                                    cur_batch_idx, [2, 1, 0], ...
                                ]
                            else:
                                video = input_pathway[cur_batch_idx]
                            # Permute to (T, H, W, C) from (C, T, H, W).
                            video = video.permute(1, 2, 3, 0)
                            video = data_utils.revert_tensor_normalize(
                                video.cpu(), cfg.DATA.MEAN, cfg.DATA.STD
                            )
                            bboxes = (
                                None
                                if cur_boxes is None
                                else cur_boxes[:, 1:].cpu()
                            )
                            video = video_vis.draw_clip(
                                video, cur_preds, bboxes=bboxes
                            )
                            video = (
                                torch.Tensor(video)
                                .permute(0, 3, 1, 2)
                                .unsqueeze(0)
                            )
                            writer.add_video(
                                video,
                                tag="Input {}/Input from pathway {}".format(
                                    global_idx, path_idx + 1
                                ),
                            )
                    if cfg.TENSORBOARD.MODEL_VIS.ACTIVATIONS:
                        writer.plot_weights_and_activations(
                            cur_activations,
                            tag="Input {}/Activations: ".format(global_idx),
                            batch_idx=cur_batch_idx,
                            indexing_dict=indexing_dict,
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
            misc.log_model_info(model, cfg, use_train_input=False)

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
