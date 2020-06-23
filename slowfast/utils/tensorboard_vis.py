#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import logging as log
import os
import torch
from torch.utils.tensorboard import SummaryWriter

import slowfast.utils.logging as logging
import slowfast.utils.visualization_utils as vis_utils
from slowfast.utils.misc import get_class_names

logger = logging.get_logger(__name__)
log.getLogger("matplotlib").setLevel(log.ERROR)


class TensorboardWriter(object):
    """
    Helper class to log information to Tensorboard.
    """

    def __init__(self, cfg):
        """
        Args:
            cfg (CfgNode): configs. Details can be found in
                slowfast/config/defaults.py
        """
        # class_names: list of class names.
        # cm_subset_classes: a list of class ids -- a user-specified subset.
        # parent_map: dictionary where key is the parent class name and
        #   value is a list of ids of its children classes.
        # hist_subset_classes: a list of class ids -- user-specified to plot histograms.
        (
            self.class_names,
            self.cm_subset_classes,
            self.parent_map,
            self.hist_subset_classes,
        ) = (None, None, None, None)
        self.cfg = cfg
        self.cm_figsize = cfg.TENSORBOARD.CONFUSION_MATRIX.FIGSIZE
        self.hist_figsize = cfg.TENSORBOARD.HISTOGRAM.FIGSIZE

        if cfg.TENSORBOARD.LOG_DIR == "":
            log_dir = os.path.join(
                cfg.OUTPUT_DIR, "runs-{}".format(cfg.TRAIN.DATASET)
            )
        else:
            log_dir = os.path.join(cfg.OUTPUT_DIR, cfg.TENSORBOARD.LOG_DIR)

        self.writer = SummaryWriter(log_dir=log_dir)
        logger.info(
            "To see logged results in Tensorboard, please launch using the command \
            `tensorboard  --port=<port-number> --logdir {}`".format(
                log_dir
            )
        )

        if cfg.TENSORBOARD.CLASS_NAMES_PATH != "":
            if cfg.DETECTION.ENABLE:
                logger.info(
                    "Plotting confusion matrix is currently \
                    not supported for detection."
                )
            (
                self.class_names,
                self.parent_map,
                self.cm_subset_classes,
            ) = get_class_names(
                cfg.TENSORBOARD.CLASS_NAMES_PATH,
                cfg.TENSORBOARD.CATEGORIES_PATH,
                cfg.TENSORBOARD.CONFUSION_MATRIX.SUBSET_PATH,
            )

            if cfg.TENSORBOARD.HISTOGRAM.ENABLE:
                if cfg.DETECTION.ENABLE:
                    logger.info(
                        "Plotting histogram is not currently \
                    supported for detection tasks."
                    )
                if cfg.TENSORBOARD.HISTOGRAM.SUBSET_PATH != "":
                    _, _, self.hist_subset_classes = get_class_names(
                        cfg.TENSORBOARD.CLASS_NAMES_PATH,
                        None,
                        cfg.TENSORBOARD.HISTOGRAM.SUBSET_PATH,
                    )

    def add_scalars(self, data_dict, global_step=None):
        """
        Add multiple scalars to Tensorboard logs.
        Args:
            data_dict (dict): key is a string specifying the tag of value.
            global_step (Optinal[int]): Global step value to record.
        """
        if self.writer is not None:
            for key, item in data_dict.items():
                self.writer.add_scalar(key, item, global_step)

    def plot_eval(self, preds, labels, global_step=None):
        """
        Plot confusion matrices and histograms for eval/test set.
        Args:
            preds (tensor or list of tensors): list of predictions.
            labels (tensor or list of tensors): list of labels.
            global step (Optional[int]): current step in eval/test.
        """
        if not self.cfg.DETECTION.ENABLE:
            cmtx = None
            if self.cfg.TENSORBOARD.CONFUSION_MATRIX.ENABLE:
                cmtx = vis_utils.get_confusion_matrix(
                    preds, labels, self.cfg.MODEL.NUM_CLASSES
                )
                # Add full confusion matrix.
                add_confusion_matrix(
                    self.writer,
                    cmtx,
                    self.cfg.MODEL.NUM_CLASSES,
                    global_step=global_step,
                    class_names=self.class_names,
                    figsize=self.cm_figsize,
                )
                # If a list of subset is provided, plot confusion matrix subset.
                if self.cm_subset_classes is not None:
                    add_confusion_matrix(
                        self.writer,
                        cmtx,
                        self.cfg.MODEL.NUM_CLASSES,
                        global_step=global_step,
                        subset_ids=self.cm_subset_classes,
                        class_names=self.class_names,
                        tag="Confusion Matrix Subset",
                        figsize=self.cm_figsize,
                    )
                # If a parent-child classes mapping is provided, plot confusion
                # matrices grouped by parent classes.
                if self.parent_map is not None:
                    # Get list of tags (parent categories names) and their children.
                    for parent_class, children_ls in self.parent_map.items():
                        tag = (
                            "Confusion Matrices Grouped by Parent Classes/"
                            + parent_class
                        )
                        add_confusion_matrix(
                            self.writer,
                            cmtx,
                            self.cfg.MODEL.NUM_CLASSES,
                            global_step=global_step,
                            subset_ids=children_ls,
                            class_names=self.class_names,
                            tag=tag,
                            figsize=self.cm_figsize,
                        )
            if self.cfg.TENSORBOARD.HISTOGRAM.ENABLE:
                if cmtx is None:
                    cmtx = vis_utils.get_confusion_matrix(
                        preds, labels, self.cfg.MODEL.NUM_CLASSES
                    )
                plot_hist(
                    self.writer,
                    cmtx,
                    self.cfg.MODEL.NUM_CLASSES,
                    self.cfg.TENSORBOARD.HISTOGRAM.TOPK,
                    global_step=global_step,
                    subset_ids=self.hist_subset_classes,
                    class_names=self.class_names,
                    figsize=self.hist_figsize,
                )

    def close(self):
        self.writer.flush()
        self.writer.close()


def add_confusion_matrix(
    writer,
    cmtx,
    num_classes,
    global_step=None,
    subset_ids=None,
    class_names=None,
    tag="Confusion Matrix",
    figsize=None,
):
    """
    Calculate and plot confusion matrix to a SummaryWriter.
    Args:
        writer (SummaryWriter): the SummaryWriter to write the matrix to.
        cmtx (ndarray): confusion matrix.
        num_classes (int): total number of classes.
        global_step (Optional[int]): current step.
        subset_ids (list of ints): a list of label indices to keep.
        class_names (list of strs, optional): a list of all class names.
        tag (str or list of strs): name(s) of the confusion matrix image.
        figsize (Optional[float, float]): the figure size of the confusion matrix.
            If None, default to [6.4, 4.8].

    """
    if subset_ids is None or len(subset_ids) != 0:
        # If class names are not provided, use class indices as class names.
        if class_names is None:
            class_names = [str(i) for i in range(num_classes)]
        # If subset is not provided, take every classes.
        if subset_ids is None:
            subset_ids = list(range(num_classes))

        sub_cmtx = cmtx[subset_ids, :][:, subset_ids]
        sub_names = [class_names[j] for j in subset_ids]

        sub_cmtx = vis_utils.plot_confusion_matrix(
            sub_cmtx, num_classes=len(subset_ids), class_names=sub_names, figsize=figsize,
        )
        # Add the confusion matrix image to writer.
        writer.add_figure(tag=tag, figure=sub_cmtx, global_step=global_step)


def plot_hist(
    writer,
    cmtx,
    num_classes,
    k=10,
    global_step=None,
    subset_ids=None,
    class_names=None,
    figsize=None,
):
    """
    Given all predictions and all true labels, plot histograms of top-k most
    frequently predicted classes for each true class.

    Args:
        writer (SummaryWriter object): a tensorboard SummaryWriter object.
        cmtx (ndarray): confusion matrix.
        num_classes (int): total number of classes.
        k (int): top k to plot histograms.
        global_step (int): current step.
        subset_ids (list of ints, optional): class indices to plot histogram.
        mapping (list of strings): names of all classes.
        figsize (Optional[float, float]): the figure size of the confusion matrix.
            If None, default to [6.4, 4.8].
    """
    if subset_ids is None or len(subset_ids) != 0:
        if subset_ids is None:
            subset_ids = set(range(num_classes))
        else:
            subset_ids = set(subset_ids)
        # If class names are not provided, use their indices as names.
        if class_names is None:
            class_names = list(range(num_classes))

        for i in subset_ids:
            pred = cmtx[i]
            hist = vis_utils.plot_topk_histogram(
                class_names[i], torch.Tensor(pred), k, class_names, figsize=figsize
            )
            writer.add_figure(
                tag="Top {} predictions by classes/{}".format(
                    k, class_names[i]
                ),
                figure=hist,
                global_step=global_step,
            )
