#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import numpy as np
from time import time
import pandas as pd
import torch
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor

import slowfast.utils.checkpoint as cu
import slowfast.utils.distributed as du
from slowfast.datasets import cv2_transform
from slowfast.datasets.cv2_transform import scale
from slowfast.datasets.utils import tensor_normalize
from slowfast.models import build
from slowfast.utils import logging, misc

import cv2

logger = logging.get_logger(__name__)
np.random.seed(20)


class VideoReader(object):
    def __init__(self, cfg):
        self.source = cfg.DEMO.DATA_SOURCE
        self.display_width = cfg.DEMO.DISPLAY_WIDTH
        self.display_height = cfg.DEMO.DISPLAY_HEIGHT
        try:  # OpenCV needs int to read from webcam.
            self.source = int(self.source)
        except ValueError:
            pass
        self.cap = cv2.VideoCapture(self.source)
        if self.display_width > 0 and self.display_height > 0:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.display_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.display_height)
        else:
            self.display_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.display_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if not self.cap.isOpened():
            raise IOError("Video {} cannot be opened".format(self.source))
        self.output_file = None
        if cfg.DEMO.OUTPUT_FILE != "":
            self.output_file = self.get_output_file(cfg.DEMO.OUTPUT_FILE)

    def __iter__(self):
        return self

    def __next__(self):
        was_read, frame = self.cap.read()
        if not was_read:
            ## reiterate the video instead of quiting.
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frame = None

        return was_read, frame

    def get_output_file(self, path):
        return cv2.VideoWriter(
            filename=path,
            fourcc=cv2.VideoWriter_fourcc(*"mp4v"),
            fps=float(30),
            frameSize=(self.display_width, self.display_height),
            isColor=True,
        )

    def display(self, frame):
        if self.output_file is None:
            cv2.imshow("SlowFast", frame)
        else:
            self.output_file.write(frame)

    def clean(self):
        self.cap.release()
        if self.output_file is None:
            cv2.destroyAllWindows()
        else:
            self.output_file.release()


def demo(cfg):
    """
    Run inference on an input video or stream from webcam.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    """
    # Set random seed from configs.
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)

    # Setup logging format.
    logging.setup_logging()

    # Print config.
    logger.info("Run demo with config:")
    logger.info(cfg)
    # Build the video model and print model statistics.
    model = build.build_model(cfg)
    model.eval()
    misc.log_model_info(model, cfg)

    # Load a checkpoint to test if applicable.
    if cfg.TEST.CHECKPOINT_FILE_PATH != "":
        ckpt = cfg.TEST.CHECKPOINT_FILE_PATH
    elif cu.has_checkpoint(cfg.OUTPUT_DIR):
        ckpt = cu.get_last_checkpoint(cfg.OUTPUT_DIR)
    elif cfg.TRAIN.CHECKPOINT_FILE_PATH != "":
        # If no checkpoint found in TEST.CHECKPOINT_FILE_PATH or in the current
        # checkpoint folder, try to load checkpoint from
        # TRAIN.CHECKPOINT_FILE_PATH and test it.
        ckpt = cfg.TRAIN.CHECKPOINT_FILE_PATH
    else:
        raise NotImplementedError("Unknown way to load checkpoint.")

    cu.load_checkpoint(
        ckpt,
        model,
        cfg.NUM_GPUS > 1,
        None,
        inflation=False,
        convert_from_caffe2="caffe2"
        in [cfg.TEST.CHECKPOINT_TYPE, cfg.TRAIN.CHECKPOINT_TYPE],
    )

    if cfg.DETECTION.ENABLE:
        # Load object detector from detectron2.
        dtron2_cfg_file = cfg.DEMO.DETECTRON2_OBJECT_DETECTION_MODEL_CFG
        dtron2_cfg = get_cfg()
        dtron2_cfg.merge_from_file(model_zoo.get_config_file(dtron2_cfg_file))
        dtron2_cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        dtron2_cfg.MODEL.WEIGHTS = (
            cfg.DEMO.DETECTRON2_OBJECT_DETECTION_MODEL_WEIGHTS
        )
        logger.info("Initialize detectron2 model.")
        object_predictor = DefaultPredictor(dtron2_cfg)
        # Load the labels of AVA dataset
        with open(cfg.DEMO.LABEL_FILE_PATH) as f:
            labels = f.read().split("\n")[:-1]
        palette = np.random.randint(64, 128, (len(labels), 3)).tolist()
        boxes = []
        logger.info("Finish loading detectron2")
    else:
        # Load the labels of Kinectics-400 dataset.
        labels_df = pd.read_csv(cfg.DEMO.LABEL_FILE_PATH)
        labels = labels_df["name"].values

    frame_provider = VideoReader(cfg)

    seq_len = cfg.DATA.NUM_FRAMES * cfg.DATA.SAMPLING_RATE
    frames = []
    pred_labels = []
    s = 0.0
    for able_to_read, frame in frame_provider:
        if not able_to_read:
            # when reaches the end frame, clear the buffer and continue to the next one.
            frames = []
            break

        if len(frames) != seq_len:
            frame_processed = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_processed = scale(cfg.DATA.TEST_CROP_SIZE, frame_processed)
            frames.append(frame_processed)
            if cfg.DETECTION.ENABLE and len(frames) == seq_len // 2 - 1:
                mid_frame = frame

        if len(frames) == seq_len:
            start = time()
            if cfg.DETECTION.ENABLE:
                outputs = object_predictor(mid_frame)
                fields = outputs["instances"]._fields
                pred_classes = fields["pred_classes"]
                selection_mask = pred_classes == 0
                # acquire person boxes.
                pred_classes = pred_classes[selection_mask]
                pred_boxes = fields["pred_boxes"].tensor[selection_mask]
                boxes = cv2_transform.scale_boxes(
                    cfg.DATA.TEST_CROP_SIZE,
                    pred_boxes,
                    frame_provider.display_height,
                    frame_provider.display_width,
                )
                boxes = torch.cat(
                    [torch.full((boxes.shape[0], 1), float(0)).cuda(), boxes],
                    axis=1,
                )
            inputs = tensor_normalize(
                torch.as_tensor(frames), cfg.DATA.MEAN, cfg.DATA.STD
            )

            # T H W C -> C T H W.
            inputs = inputs.permute(3, 0, 1, 2)

            # 1 C T H W.
            inputs = inputs.unsqueeze(0)
            if cfg.MODEL.ARCH in cfg.MODEL.SINGLE_PATHWAY_ARCH:
                # Sample frames for the fast pathway.
                index = torch.linspace(
                    0, inputs.shape[2] - 1, cfg.DATA.NUM_FRAMES
                ).long()
                inputs = [torch.index_select(inputs, 2, index)]
            elif cfg.MODEL.ARCH in cfg.MODEL.MULTI_PATHWAY_ARCH:
                # Sample frames for the fast pathway.
                index = torch.linspace(
                    0, inputs.shape[2] - 1, cfg.DATA.NUM_FRAMES
                ).long()
                fast_pathway = torch.index_select(inputs, 2, index)

                # Sample frames for the slow pathway.
                index = torch.linspace(
                    0,
                    fast_pathway.shape[2] - 1,
                    fast_pathway.shape[2] // cfg.SLOWFAST.ALPHA,
                ).long()
                slow_pathway = torch.index_select(fast_pathway, 2, index)
                inputs = [slow_pathway, fast_pathway]
            else:
                raise NotImplementedError(
                    "Model arch {} is not in {}".format(
                        cfg.MODEL.ARCH,
                        cfg.MODEL.SINGLE_PATHWAY_ARCH
                        + cfg.MODEL.MULTI_PATHWAY_ARCH,
                    )
                )

            # Transfer the data to the current GPU device.
            if isinstance(inputs, (list,)):
                for i in range(len(inputs)):
                    inputs[i] = inputs[i].cuda(non_blocking=True)
            else:
                inputs = inputs.cuda(non_blocking=True)

            # Perform the forward pass.
            if cfg.DETECTION.ENABLE:
                # When there is nothing in the scene,
                #   use a dummy variable to disable all computations below.
                if not len(boxes):
                    preds = torch.tensor([])
                else:
                    preds = model(inputs, boxes)
            else:
                preds = model(inputs)

            # Gather all the predictions across all the devices to perform ensemble.
            if cfg.NUM_GPUS > 1:
                preds = du.all_gather(preds)[0]

            if cfg.DETECTION.ENABLE:
                # This post processing was intendedly assigned to the cpu since my laptop GPU
                #   RTX 2080 runs out of its memory, if your GPU is more powerful, I'd recommend
                #   to change this section to make CUDA does the processing.
                preds = preds.cpu().detach().numpy()
                pred_masks = preds > 0.1
                label_ids = [
                    np.nonzero(pred_mask)[0] for pred_mask in pred_masks
                ]
                pred_labels = [
                    [labels[label_id] for label_id in perbox_label_ids]
                    for perbox_label_ids in label_ids
                ]
                # I'm unsure how to detectron2 rescales boxes to image original size, so I use
                #   input boxes of slowfast and rescale back it instead, it's safer and even if boxes
                #   was not rescaled by cv2_transform.rescale_boxes, it still works.
                boxes = boxes.cpu().detach().numpy()
                ratio = (
                    np.min(
                        [
                            frame_provider.display_height,
                            frame_provider.display_width,
                        ]
                    )
                    / cfg.DATA.TEST_CROP_SIZE
                )
                boxes = boxes[:, 1:] * ratio
            else:
                ## Option 1: single label inference selected from the highest probability entry.
                # label_id = preds.argmax(-1).cpu()
                # pred_label = labels[label_id]
                # Option 2: multi-label inferencing selected from probability entries > threshold.
                label_ids = (
                    torch.nonzero(preds.squeeze() > 0.1)
                    .reshape(-1)
                    .cpu()
                    .detach()
                    .numpy()
                )
                pred_labels = labels[label_ids]
                logger.info(pred_labels)
                if not list(pred_labels):
                    pred_labels = ["Unknown"]

            # # option 1: remove the oldest frame in the buffer to make place for the new one.
            # frames.pop(0)
            # option 2: empty the buffer
            frames = []
            s = time() - start

        if cfg.DETECTION.ENABLE and pred_labels and boxes.any():
            for box, box_labels in zip(boxes.astype(int), pred_labels):
                cv2.rectangle(
                    frame,
                    tuple(box[:2]),
                    tuple(box[2:]),
                    (0, 255, 0),
                    thickness=2,
                )
                label_origin = box[:2]
                for label in box_labels:
                    label_origin[-1] -= 5
                    (label_width, label_height), _ = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
                    )
                    cv2.rectangle(
                        frame,
                        (label_origin[0], label_origin[1] + 5),
                        (
                            label_origin[0] + label_width,
                            label_origin[1] - label_height - 5,
                        ),
                        palette[labels.index(label)],
                        -1,
                    )
                    cv2.putText(
                        frame,
                        label,
                        tuple(label_origin),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        1,
                    )
                    label_origin[-1] -= label_height + 5
        if not cfg.DETECTION.ENABLE:
            # Display predicted labels to frame.
            y_offset = 50
            cv2.putText(
                frame,
                "Action:",
                (10, y_offset),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.65,
                color=(0, 235, 0),
                thickness=2,
            )
            for pred_label in pred_labels:
                y_offset += 30
                cv2.putText(
                    frame,
                    "{}".format(pred_label),
                    (20, y_offset),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.65,
                    color=(0, 235, 0),
                    thickness=2,
                )

        # Display prediction speed.
        cv2.putText(
            frame,
            "Speed: {:.2f}s".format(s),
            (10, 25),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.65,
            color=(0, 235, 0),
            thickness=2,
        )
        frame_provider.display(frame)
        # hit Esc to quit the demo.
        key = cv2.waitKey(1)
        if key == 27:
            break

    frame_provider.clean()
