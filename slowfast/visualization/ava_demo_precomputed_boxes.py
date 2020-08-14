#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import numpy as np
import os
import cv2
import torch
import tqdm
from fvcore.common.file_io import PathManager

import slowfast.utils.checkpoint as cu
import slowfast.utils.logging as logging
from slowfast.datasets.ava_helper import parse_bboxes_file
from slowfast.datasets.cv2_transform import scale, scale_boxes
from slowfast.datasets.utils import get_sequence
from slowfast.models import build_model
from slowfast.utils import misc
from slowfast.visualization.utils import process_cv2_inputs
from slowfast.visualization.video_visualizer import VideoVisualizer

logger = logging.get_logger(__name__)


class AVAVisualizerWithPrecomputedBox:
    """
    Visualize action predictions for videos or folder of images with precomputed
    and ground-truth boxes in AVA format.
    """

    def __init__(self, cfg):
        """
        Args:
            cfg (CfgNode): configs. Details can be found in
                slowfast/config/defaults.py
        """
        self.source = PathManager.get_local_path(path=cfg.DEMO.INPUT_VIDEO)
        self.fps = None
        if PathManager.isdir(self.source):
            self.fps = cfg.DEMO.FPS
            self.video_name = self.source.split("/")[-1]
            self.source = os.path.join(
                self.source, "{}_%06d.jpg".format(self.video_name)
            )
        else:
            self.video_name = self.source.split("/")[-1]
            self.video_name = self.video_name.split(".")[0]

        self.cfg = cfg
        self.cap = cv2.VideoCapture(self.source)
        if self.fps is None:
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)

        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        self.display_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.display_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if not self.cap.isOpened():
            raise IOError("Video {} cannot be opened".format(self.source))

        self.output_file = None

        if cfg.DEMO.OUTPUT_FILE != "":
            self.output_file = self.get_output_file(cfg.DEMO.OUTPUT_FILE)

        self.pred_boxes, self.gt_boxes = load_boxes_labels(
            cfg,
            self.video_name,
            self.fps,
            self.display_width,
            self.display_height,
        )

        self.seq_length = cfg.DATA.NUM_FRAMES * cfg.DATA.SAMPLING_RATE
        self.no_frames_repeat = cfg.DEMO.SLOWMO

    def get_output_file(self, path):
        """
        Return a video writer object.
        Args:
            path (str): path to the output video file.
        """
        return cv2.VideoWriter(
            filename=path,
            fourcc=cv2.VideoWriter_fourcc(*"mp4v"),
            fps=float(30),
            frameSize=(self.display_width, self.display_height),
            isColor=True,
        )

    def get_input_clip(self, keyframe_idx):
        """
        Get input clip from the video/folder of images for a given
        keyframe index.
        Args:
            keyframe_idx (int): index of the current keyframe.
        Returns:
            clip (list of tensors): formatted input clip(s) corresponding to
                the current keyframe.
        """
        seq = get_sequence(
            keyframe_idx,
            self.seq_length // 2,
            self.cfg.DATA.SAMPLING_RATE,
            self.total_frames,
        )
        clip = []
        for frame_idx in seq:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            was_read, frame = self.cap.read()
            if was_read:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = scale(self.cfg.DATA.TEST_CROP_SIZE, frame)
                clip.append(frame)
            else:
                logger.error(
                    "Unable to read frame. Duplicating previous frame."
                )
                clip.append(clip[-1])

        clip = process_cv2_inputs(clip, self.cfg)
        return clip

    def get_predictions(self):
        """
        Predict and append prediction results to each box in each keyframe in
        `self.pred_boxes` dictionary.
        """
        # Set random seed from configs.
        np.random.seed(self.cfg.RNG_SEED)
        torch.manual_seed(self.cfg.RNG_SEED)

        # Setup logging format.
        logging.setup_logging(self.cfg.OUTPUT_DIR)

        # Print config.
        logger.info("Run demo with config:")
        logger.info(self.cfg)
        assert (
            self.cfg.NUM_GPUS <= 1
        ), "Cannot run demo visualization on multiple GPUs."

        # Build the video model and print model statistics.
        model = build_model(self.cfg)
        model.eval()
        logger.info("Start loading model info")
        misc.log_model_info(model, self.cfg, use_train_input=False)
        logger.info("Start loading model weights")
        cu.load_test_checkpoint(self.cfg, model)
        logger.info("Finish loading model weights")
        logger.info("Start making predictions for precomputed boxes.")
        for keyframe_idx, boxes_and_labels in tqdm.tqdm(
            self.pred_boxes.items()
        ):
            inputs = self.get_input_clip(keyframe_idx)
            boxes = boxes_and_labels[0]
            boxes = torch.from_numpy(np.array(boxes)).float()

            box_transformed = scale_boxes(
                self.cfg.DATA.TEST_CROP_SIZE,
                boxes,
                self.display_height,
                self.display_width,
            )

            # Pad frame index for each box.
            box_inputs = torch.cat(
                [
                    torch.full((box_transformed.shape[0], 1), float(0)),
                    box_transformed,
                ],
                axis=1,
            )
            if self.cfg.NUM_GPUS:
                # Transfer the data to the current GPU device.
                if isinstance(inputs, (list,)):
                    for i in range(len(inputs)):
                        inputs[i] = inputs[i].cuda(non_blocking=True)
                else:
                    inputs = inputs.cuda(non_blocking=True)

                box_inputs = box_inputs.cuda()

            preds = model(inputs, box_inputs)

            preds = preds.detach()

            if self.cfg.NUM_GPUS:
                preds = preds.cpu()

            boxes_and_labels[1] = preds

    def draw_video(self):
        """
        Draw predicted and ground-truth (if provided) results on the video/folder of images.
        Write the visualized result to a video output file.
        """
        all_boxes = merge_pred_gt_boxes(self.pred_boxes, self.gt_boxes)
        common_classes = (
            self.cfg.DEMO.COMMON_CLASS_NAMES
            if len(self.cfg.DEMO.LABEL_FILE_PATH) != 0
            else None
        )
        video_vis = VideoVisualizer(
            num_classes=self.cfg.MODEL.NUM_CLASSES,
            class_names_path=self.cfg.DEMO.LABEL_FILE_PATH,
            top_k=self.cfg.TENSORBOARD.MODEL_VIS.TOPK_PREDS,
            thres=self.cfg.DEMO.COMMON_CLASS_THRES,
            lower_thres=self.cfg.DEMO.UNCOMMON_CLASS_THRES,
            common_class_names=common_classes,
            colormap=self.cfg.TENSORBOARD.MODEL_VIS.COLORMAP,
            mode=self.cfg.DEMO.VIS_MODE,
        )

        all_keys = sorted(all_boxes.keys())
        # Draw around the keyframe for 2/10 of the sequence length.
        # This is chosen using heuristics.
        draw_range = [
            self.seq_length // 2 - self.seq_length // 10,
            self.seq_length // 2 + self.seq_length // 10,
        ]
        draw_range_repeat = [
            draw_range[0],
            (draw_range[1] - draw_range[0]) * self.no_frames_repeat
            + draw_range[0],
        ]
        prev_buffer = []
        prev_end_idx = 0

        logger.info("Start Visualization...")
        for keyframe_idx in tqdm.tqdm(all_keys):
            pred_gt_boxes = all_boxes[keyframe_idx]
            # Find the starting index of the clip. If start_idx exceeds the beginning
            # of the video, we only choose valid frame from index 0.
            start_idx = max(0, keyframe_idx - self.seq_length // 2)
            # Number of frames from the start of the current clip and the
            # end of the previous clip.
            dist = start_idx - prev_end_idx
            # If there are unwritten frames in between clips.
            if dist >= 0:
                # Get the frames in between previous clip and current clip.
                frames = self._get_frame_range(prev_end_idx, dist)
                # We keep a buffer of frames for overlapping visualization.
                # Write these to the output file.
                for frame in prev_buffer:
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    self.display(frame)
                # Write them to output file without any visualization
                # since they don't have any corresponding keyframes.
                for frame in frames:
                    self.display(frame)
                prev_buffer = []
                num_new_frames = self.seq_length

            # If there are overlapping frames in between clips.
            elif dist < 0:
                # Flush all ready frames.
                for frame in prev_buffer[:dist]:
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    self.display(frame)
                prev_buffer = prev_buffer[dist:]
                num_new_frames = self.seq_length + dist
            # Obtain new frames for the current clip from the input video file.
            new_frames = self._get_frame_range(
                max(start_idx, prev_end_idx), num_new_frames
            )
            new_frames = [
                cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in new_frames
            ]
            clip = prev_buffer + new_frames
            # Calculate the end of this clip. This will be `prev_end_idx` for the
            # next iteration.
            prev_end_idx = max(start_idx, prev_end_idx) + len(new_frames)
            # For each precomputed or gt boxes.
            for i, boxes in enumerate(pred_gt_boxes):
                if i == 0:
                    repeat = self.no_frames_repeat
                    current_draw_range = draw_range
                else:
                    repeat = 1
                    current_draw_range = draw_range_repeat
                # Make sure draw range does not fall out of end of clip.
                current_draw_range[1] = min(
                    current_draw_range[1], len(clip) - 1
                )
                ground_truth = boxes[0]
                bboxes = boxes[1]
                label = boxes[2]
                # Draw predictions.
                clip = video_vis.draw_clip_range(
                    clip,
                    label,
                    bboxes=torch.Tensor(bboxes),
                    ground_truth=ground_truth,
                    draw_range=current_draw_range,
                    repeat_frame=repeat,
                )
            # Store the current clip as buffer.
            prev_buffer = clip

        # Write the remaining buffer to output file.
        for frame in prev_buffer:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            self.display(frame)
        # If we still have some remaining frames in the input file,
        # write those to the output file as well.
        if prev_end_idx < self.total_frames:
            dist = self.total_frames - prev_end_idx
            remaining_clip = self._get_frame_range(prev_end_idx, dist)
            for frame in remaining_clip:
                self.display(frame)

    def __call__(self):
        self.get_predictions()
        self.draw_video()

    def display(self, frame):
        """
        Either display a single frame (BGR image) to a window or write to
        an output file if output path is provided.
        """
        if self.output_file is None:
            cv2.imshow("SlowFast", frame)
        else:
            self.output_file.write(frame)

    def _get_keyframe_clip(self, keyframe_idx):
        """
        Return a clip corresponding to a keyframe index for visualization.
        Args:
            keyframe_idx (int): keyframe index.
        """
        start_idx = max(0, keyframe_idx - self.seq_length // 2)

        clip = self._get_frame_range(start_idx, self.seq_length)

        return clip

    def _get_frame_range(self, start_idx, num_frames):
        """
        Return a clip of `num_frames` frames starting from `start_idx`. If not enough frames
        from `start_idx`, return the remaining frames from `start_idx`.
        Args:
            start_idx (int): starting idx.
            num_frames (int): number of frames in the returned clip.
        """
        was_read = True
        assert start_idx < self.total_frames, "Start index out of range."

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)
        all_frames = []
        for _ in range(num_frames):
            was_read, frame = self.cap.read()
            if was_read:
                all_frames.append(frame)
            else:
                break

        return all_frames


def merge_pred_gt_boxes(pred_dict, gt_dict=None):
    """
    Merge data from precomputed and ground-truth boxes dictionaries.
    Args:
        pred_dict (dict): a dict which maps from `frame_idx` to a list of `boxes`
            and `labels`. Each `box` is a list of 4 box coordinates. `labels[i]` is
            a list of labels for `boxes[i]`.
        gt_dict (Optional[dict]): a dict which maps from `frame_idx` to a list of `boxes`
            and `labels`. Each `box` is a list of 4 box coordinates. `labels[i]` is
            a list of labels for `boxes[i]`. Note that label is -1 for predicted boxes.
    Returns:
        merged_dict (dict): merged dictionary from `pred_dict` and `gt_dict` if given.
            It is a dict which maps from `frame_idx` to a list of [`is_gt`, `boxes`, `labels`],
            where `is_gt` is a boolean indicate whether the `boxes` and `labels` are ground-truth.
    """
    merged_dict = {}
    for key, item in pred_dict.items():
        merged_dict[key] = [[False, item[0], item[1]]]

    if gt_dict is not None:
        for key, item in gt_dict.items():
            if merged_dict.get(key) is None:
                merged_dict[key] = [[True, item[0], item[1]]]
            else:
                merged_dict[key].append([True, item[0], item[1]])
    return merged_dict


def load_boxes_labels(cfg, video_name, fps, img_width, img_height):
    """
    Loading boxes and labels from AVA bounding boxes csv files.
    Args:
        cfg (CfgNode): config.
        video_name (str): name of the given video.
        fps (int or float): frames per second of the input video/images folder.
        img_width (int): width of images in input video/images folder.
        img_height (int): height of images in input video/images folder.
    Returns:
        preds_boxes (dict): a dict which maps from `frame_idx` to a list of `boxes`
            and `labels`. Each `box` is a list of 4 box coordinates. `labels[i]` is
            a list of labels for `boxes[i]`. Note that label is -1 for predicted boxes.
        gt_boxes (dict): if cfg.DEMO.GT_BOXES is given, return similar dict as
            all_pred_boxes but for ground-truth boxes.
    """
    starting_second = cfg.DEMO.STARTING_SECOND

    def sec_to_frameidx(sec):
        return (sec - starting_second) * fps

    def process_bboxes_dict(dictionary):
        """
        Replace all `keyframe_sec` in `dictionary` with `keyframe_idx` and
        merge all [`box_coordinate`, `box_labels`] pairs into
        [`all_boxes_coordinates`, `all_boxes_labels`] for each `keyframe_idx`.
        Args:
            dictionary (dict): a dictionary which maps `frame_sec` to a list of `box`.
                Each `box` is a [`box_coord`, `box_labels`] where `box_coord` is the
                coordinates of box and 'box_labels` are the corresponding
                labels for the box.
        Returns:
            new_dict (dict): a dict which maps from `frame_idx` to a list of `boxes`
                and `labels`. Each `box` in `boxes` is a list of 4 box coordinates. `labels[i]`
                is a list of labels for `boxes[i]`. Note that label is -1 for predicted boxes.
        """
        # Replace all keyframe_sec with keyframe_idx.
        new_dict = {}
        for keyframe_sec, boxes_and_labels in dictionary.items():
            # Ignore keyframes with no boxes
            if len(boxes_and_labels) == 0:
                continue
            keyframe_idx = sec_to_frameidx(keyframe_sec)
            boxes, labels = list(zip(*boxes_and_labels))
            # Shift labels from [1, n_classes] to [0, n_classes - 1].
            labels = [[i - 1 for i in box_label] for box_label in labels]
            boxes = np.array(boxes)
            boxes[:, [0, 2]] *= img_width
            boxes[:, [1, 3]] *= img_height
            new_dict[keyframe_idx] = [boxes.tolist(), list(labels)]
        return new_dict

    preds_boxes_path = cfg.DEMO.PREDS_BOXES
    gt_boxes_path = cfg.DEMO.GT_BOXES

    preds_boxes, _, _ = parse_bboxes_file(
        ann_filenames=[preds_boxes_path],
        ann_is_gt_box=[False],
        detect_thresh=cfg.AVA.DETECTION_SCORE_THRESH,
        boxes_sample_rate=1,
    )
    preds_boxes = preds_boxes[video_name]
    if gt_boxes_path == "":
        gt_boxes = None
    else:
        gt_boxes, _, _ = parse_bboxes_file(
            ann_filenames=[gt_boxes_path],
            ann_is_gt_box=[True],
            detect_thresh=cfg.AVA.DETECTION_SCORE_THRESH,
            boxes_sample_rate=1,
        )
        gt_boxes = gt_boxes[video_name]

    preds_boxes = process_bboxes_dict(preds_boxes)
    if gt_boxes is not None:
        gt_boxes = process_bboxes_dict(gt_boxes)

    return preds_boxes, gt_boxes
