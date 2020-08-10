#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import numpy as np
import queue
import cv2
import torch
import tqdm

from slowfast.utils import logging
from slowfast.visualization.async_predictor import (
    AsycnActionPredictor,
    AsyncVis,
)
from slowfast.visualization.demo_loader import VideoReader
from slowfast.visualization.ava_demo_precomputed_boxes import AVAVisualizerWithPrecomputedBox
from slowfast.visualization.predictor import ActionPredictor
from slowfast.visualization.video_visualizer import VideoVisualizer

logger = logging.get_logger(__name__)


def run_demo(cfg, frame_provider):
    """
    Run demo visualization.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        frame_provider (iterator): Python iterator that return task objects that are filled
            with necessary information such as `frames`, `id` and `num_buffer_frames` for the
            prediction and visualization pipeline.
    """
    # Set random seed from configs.
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)
    # Setup logging format.
    logging.setup_logging(cfg.OUTPUT_DIR)
    # Print config.
    logger.info("Run demo with config:")
    logger.info(cfg)

    video_vis = VideoVisualizer(
        cfg.MODEL.NUM_CLASSES,
        cfg.DEMO.LABEL_FILE_PATH,
        cfg.TENSORBOARD.MODEL_VIS.TOPK_PREDS,
        cfg.TENSORBOARD.MODEL_VIS.COLORMAP,
    )
    async_vis = AsyncVis(video_vis, n_workers=cfg.DEMO.NUM_VIS_INSTANCES)

    if cfg.NUM_GPUS <= 1:
        model = ActionPredictor(cfg=cfg, async_vis=async_vis)
    else:
        model = AsycnActionPredictor(cfg, async_vis.task_queue)

    seq_len = cfg.DATA.NUM_FRAMES * cfg.DATA.SAMPLING_RATE

    assert (
        cfg.DEMO.BUFFER_SIZE <= seq_len // 2
    ), "Buffer size cannot be greater than half of sequence length."
    num_task = 0
    for able_to_read, task in frame_provider:
        if not able_to_read:
            break
        num_task += 1

        model.put(task)

        try:
            frames = async_vis.get()
            num_task -= 1
            yield frames
        except queue.Empty:
            continue
        # hit Esc to quit the demo.
        key = cv2.waitKey(1)
        if key == 27:
            break

    while num_task != 0:
        try:
            frames = async_vis.get()
            num_task -= 1
            yield frames
        except queue.Empty:
            continue
        # hit Esc to quit the demo.
        key = cv2.waitKey(1)
        if key == 27:
            break


def demo(cfg):
    """
    Run inference on an input video or stream from webcam.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    """
    # AVA format-specific visualization with precomputed boxes.
    if cfg.DETECTION.ENABLE and cfg.DEMO.PREDS_BOXES != "":
        precomputed_box_vis = AVAVisualizerWithPrecomputedBox(cfg)
        precomputed_box_vis()
    else:
        frame_provider = VideoReader(cfg)

        for frames in tqdm.tqdm(run_demo(cfg, frame_provider)):
            for frame in frames:
                frame_provider.display(frame)
        frame_provider.clean()
