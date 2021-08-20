#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import numpy as np
import torch

import slowfast.datasets.utils as data_utils
import slowfast.utils.logging as logging
import slowfast.visualization.tensorboard_vis as tb
from slowfast.utils.misc import get_class_names
from slowfast.visualization.video_visualizer import VideoVisualizer

logger = logging.get_logger(__name__)


class WrongPredictionVis:
    """
    WrongPredictionVis class for visualizing video inputs to Tensorboard
    for instances that the model makes wrong predictions.
    """

    def __init__(self, cfg):
        """
        Args:
            cfg (CfgNode): configs. Details can be found in
                slowfast/config/defaults.py
        """
        self.cfg = cfg
        self.class_names, _, self.subset = get_class_names(
            cfg.TENSORBOARD.CLASS_NAMES_PATH,
            subset_path=cfg.TENSORBOARD.WRONG_PRED_VIS.SUBSET_PATH,
        )
        if self.subset is not None:
            self.subset = set(self.subset)
        self.num_class = cfg.MODEL.NUM_CLASSES
        self.video_vis = VideoVisualizer(
            cfg.MODEL.NUM_CLASSES,
            cfg.TENSORBOARD.CLASS_NAMES_PATH,
            1,
            cfg.TENSORBOARD.MODEL_VIS.COLORMAP,
        )
        self.tag = cfg.TENSORBOARD.WRONG_PRED_VIS.TAG
        self.writer = tb.TensorboardWriter(cfg)
        self.model_incorrect_classes = set()

    def _pick_wrong_preds(self, labels, preds):
        """
        Returns a 1D tensor that contains the indices of instances that have
        wrong predictions, where true labels in in the specified subset.
        Args:
            labels (tensor): tensor of shape (n_instances,) containing class ids.
            preds (tensor): class scores from model, shape (n_intances, n_classes)
        Returns:
            mask (tensor): boolean tensor. `mask[i]` is True if `model` makes a wrong prediction.
        """
        subset_mask = torch.ones(size=(len(labels),), dtype=torch.bool)
        if self.subset is not None:
            for i, label in enumerate(labels):
                if label not in self.subset:
                    subset_mask[i] = False

        preds_ids = torch.argmax(preds, dim=-1)

        mask = preds_ids != labels
        mask &= subset_mask
        for i, wrong_pred in enumerate(mask):
            if wrong_pred:
                self.model_incorrect_classes.add(labels[i])

        return mask

    def visualize_vid(self, video_input, labels, preds, batch_idx):
        """
        Draw predicted labels on video inputs and visualize all incorrectly classified
        videos in the current batch.
        Args:
            video_input (list of list of tensor(s)): list of videos for all pathways.
            labels (array-like): shape (n_instances,) of true label for each instance.
            preds (tensor): shape (n, instances, n_classes). The predicted scores for all instances.
            tag (Optional[str]): all visualized video will be added under this tag. This is for organization
                purposes in Tensorboard.
            batch_idx (int): batch index of the current videos.
        """

        def add_video(vid, preds, tag, true_class_name):
            """
            Draw predicted label on video and add it to Tensorboard.
            Args:
                vid (array-like): shape (C, T, H, W). Each image in `vid` is a RGB image.
                preds (tensor): shape (n_classes,) or (1, n_classes). The predicted scores
                    for the current `vid`.
                tag (str): tag for `vid` in Tensorboard.
                true_class_name (str): the ground-truth class name of the current `vid` instance.
            """
            # Permute to (T, H, W, C).
            vid = vid.permute(1, 2, 3, 0)
            vid = data_utils.revert_tensor_normalize(
                vid.cpu(), self.cfg.DATA.MEAN, self.cfg.DATA.STD
            )
            vid = self.video_vis.draw_clip(vid, preds)
            vid = torch.from_numpy(np.array(vid)).permute(0, 3, 1, 2)
            vid = torch.unsqueeze(vid, dim=0)
            self.writer.add_video(
                vid, tag="{}: {}".format(tag, true_class_name)
            )

        mask = self._pick_wrong_preds(labels, preds)
        video_indices = torch.squeeze(mask.nonzero(), dim=-1)
        # Visualize each wrongly classfied video.
        for vid_idx in video_indices:
            cur_vid_idx = batch_idx * len(video_input[0]) + vid_idx
            for pathway in range(len(video_input)):
                add_video(
                    video_input[pathway][vid_idx],
                    preds=preds[vid_idx],
                    tag=self.tag
                    + "/Video {}, Pathway {}".format(cur_vid_idx, pathway),
                    true_class_name=self.class_names[labels[vid_idx]],
                )

    @property
    def wrong_class_prediction(self):
        """
        Return class ids that the model predicted incorrectly.
        """
        incorrect_class_names = [
            self.class_names[i] for i in self.model_incorrect_classes
        ]
        return list(set(incorrect_class_names))

    def clean(self):
        """
        Close Tensorboard writer.
        """
        self.writer.close()
