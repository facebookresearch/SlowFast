#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import cv2
import torch
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor

import slowfast.utils.checkpoint as cu
from slowfast.datasets import cv2_transform
from slowfast.models import build_model
from slowfast.utils import logging, misc
from slowfast.visualization.utils import process_cv2_inputs

logger = logging.get_logger(__name__)


class ActionPredictor:
    """
    Action Predictor for action recognition.
    """

    def __init__(self, cfg):
        """
        Args:
            cfg (CfgNode): configs. Details can be found in
                slowfast/config/defaults.py
        """
        # Build the video model and print model statistics.
        self.model = build_model(cfg)
        self.model.eval()
        self.cfg = cfg
        logger.info("Start loading model info")
        misc.log_model_info(self.model, cfg, use_train_input=False)
        logger.info("Start loading model weights")
        cu.load_test_checkpoint(cfg, self.model)
        logger.info("Finish loading model weights")

    def __call__(self, task):
        """
        Returns the prediction results for the current task.
        Args:
            task (TaskInfo object): task object that contain
                the necessary information for action prediction. (e.g. frames, boxes)
        Returns:
            task (TaskInfo object): the same task info object but filled with
                prediction values (a tensor) and the corresponding boxes for
                action detection task.
        """
        frames, bboxes = task.frames, task.bboxes
        if bboxes is not None:
            bboxes = cv2_transform.scale_boxes(
                self.cfg.DATA.TEST_CROP_SIZE,
                bboxes,
                task.img_height,
                task.img_width,
            )
        if self.cfg.DEMO.INPUT_FORMAT == "BGR":
            frames = [
                cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames
            ]

        frames = [
            cv2_transform.scale(self.cfg.DATA.TEST_CROP_SIZE, frame)
            for frame in frames
        ]
        inputs = process_cv2_inputs(frames, self.cfg)
        if bboxes is not None:
            index_pad = torch.full(
                size=(bboxes.shape[0], 1),
                fill_value=float(0),
                device=bboxes.device,
            )

            # Pad frame index for each box.
            bboxes = torch.cat([index_pad, bboxes], axis=1)
        if self.cfg.NUM_GPUS > 0:
            # Transfer the data to the current GPU device.
            if isinstance(inputs, (list,)):
                for i in range(len(inputs)):
                    inputs[i] = inputs[i].cuda(non_blocking=True)
            else:
                inputs = inputs.cuda(non_blocking=True)
        if self.cfg.DETECTION.ENABLE and not bboxes.shape[0]:
            preds = torch.tensor([])
        else:
            preds = self.model(inputs, bboxes)

        if self.cfg.NUM_GPUS:
            preds = preds.cpu()
            if bboxes is not None:
                bboxes = bboxes.cpu()

        preds = preds.detach()

        task.add_action_preds(preds)
        if bboxes is not None:
            task.add_bboxes(bboxes[:, 1:])

        return task


class Detectron2Predictor:
    """
    Wrapper around Detectron2 to return the required predicted bounding boxes
    as a ndarray.
    """

    def __init__(self, cfg):
        """
        Args:
            cfg (CfgNode): configs. Details can be found in
                slowfast/config/defaults.py
        """

        self.cfg = get_cfg()
        self.cfg.merge_from_file(
            model_zoo.get_config_file(cfg.DEMO.DETECTRON2_CFG)
        )
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = cfg.DEMO.DETECTRON2_THRESH
        self.cfg.MODEL.WEIGHTS = cfg.DEMO.DETECTRON2_WEIGHTS
        self.cfg.INPUT.FORMAT = cfg.DEMO.INPUT_FORMAT
        self.cfg.MODEL.DEVICE = "cuda:0" if cfg.NUM_GPUS > 0 else "cpu"

        logger.info("Initialized Detectron2 Object Detection Model.")

        self.predictor = DefaultPredictor(self.cfg)

    def __call__(self, task):
        """
        Return bounding boxes predictions as a tensor.
        Args:
            task (TaskInfo object): task object that contain
                the necessary information for action prediction. (e.g. frames, boxes)
        Returns:
            task (TaskInfo object): the same task info object but filled with
                prediction values (a tensor) and the corresponding boxes for
                action detection task.
        """
        middle_frame = task.frames[len(task.frames) // 2]
        outputs = self.predictor(middle_frame)
        # Get only human instances
        mask = outputs["instances"].pred_classes == 0
        pred_boxes = outputs["instances"].pred_boxes.tensor[mask]
        task.add_bboxes(pred_boxes)

        return task


def draw_predictions(task, video_vis):
    """
    Draw prediction for the given task.
    Args:
        task (TaskInfo object): task object that contain
            the necessary information for visualization. (e.g. frames, preds)
            All attributes must lie on CPU devices.
        video_vis (VideoVisualizer object): the video visualizer object.
    Returns:
        frames (list of ndarray): visualized frames in the clip.
    """
    boxes = task.bboxes
    frames = task.frames
    preds = task.action_preds
    if boxes is not None:
        img_width = task.img_width
        img_height = task.img_height
        boxes = cv2_transform.revert_scaled_boxes(
            task.crop_size, boxes, img_height, img_width
        )

    keyframe_idx = len(frames) // 2 - task.num_buffer_frames
    draw_range = [
        keyframe_idx - task.clip_vis_size,
        keyframe_idx + task.clip_vis_size,
    ]
    frames = frames[task.num_buffer_frames :]
    if boxes is not None:
        if len(boxes) != 0:
            frames = video_vis.draw_clip_range(
                frames,
                preds,
                boxes,
                keyframe_idx=keyframe_idx,
                draw_range=draw_range,
            )
    else:
        frames = video_vis.draw_clip_range(
            frames, preds, keyframe_idx=keyframe_idx, draw_range=draw_range
        )
    del task

    return frames
