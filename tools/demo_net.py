#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import numpy as np
import cv2
import torch
import tqdm

from slowfast.utils import logging
from slowfast.visualization.demo_loader import VideoReader
from slowfast.visualization.ava_demo_precomputed_boxes import AVAVisualizerWithPrecomputedBox
from slowfast.visualization.predictor import (
    ActionPredictor,
    Detectron2Predictor,
    draw_predictions,
)
from slowfast.visualization.utils import init_task_info
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
    assert cfg.NUM_GPUS <= 1, "Cannot run demo on multiple GPUs."
    # Print config.
    logger.info("Run demo with config:")
    logger.info(cfg)
    video_vis = VideoVisualizer(
        cfg.MODEL.NUM_CLASSES,
        cfg.DEMO.LABEL_FILE_PATH,
        cfg.TENSORBOARD.MODEL_VIS.TOPK_PREDS,
        cfg.TENSORBOARD.MODEL_VIS.COLORMAP,
    )

    if cfg.DETECTION.ENABLE:
        object_detector = Detectron2Predictor(cfg)

    model = ActionPredictor(cfg)

    seq_len = cfg.DATA.NUM_FRAMES * cfg.DATA.SAMPLING_RATE
    assert (
        cfg.DEMO.BUFFER_SIZE <= seq_len // 2
    ), "Buffer size cannot be greater than half of sequence length."
    init_task_info(
        frame_provider.display_height,
        frame_provider.display_width,
        cfg.DATA.TEST_CROP_SIZE,
        cfg.DEMO.CLIP_VIS_SIZE,
    )
    for able_to_read, task in frame_provider:
        if not able_to_read:
            break

        if cfg.DETECTION.ENABLE:
            task = object_detector(task)

        task = model(task)
        frames = draw_predictions(task, video_vis)
        # hit Esc to quit the demo.
        key = cv2.waitKey(1)
        if key == 27:
            break
        yield frames


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
