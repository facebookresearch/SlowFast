#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import logging as log
import math
import os
import matplotlib.pyplot as plt
import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

import slowfast.utils.logging as logging
import slowfast.visualization.utils as vis_utils
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

    def add_video(self, vid_tensor, tag="Video Input", global_step=None, fps=4):
        """
        Add input to tensorboard SummaryWriter as a video.
        Args:
            vid_tensor (tensor): shape of (B, T, C, H, W). Values should lie
                [0, 255] for type uint8 or [0, 1] for type float.
            tag (Optional[str]): name of the video.
            global_step(Optional[int]): current step.
            fps (int): frames per second.
        """
        self.writer.add_video(tag, vid_tensor, global_step=global_step, fps=fps)

    def plot_weights_and_activations(
        self,
        weight_activation_dict,
        tag="",
        normalize=False,
        global_step=None,
        batch_idx=None,
        indexing_dict=None,
        heat_map=True,
    ):
        """
        Visualize weights/ activations tensors to Tensorboard.
        Args:
            weight_activation_dict (dict[str, tensor]): a dictionary of the pair {layer_name: tensor},
                where layer_name is a string and tensor is the weights/activations of
                the layer we want to visualize.
            tag (Optional[str]): name of the video.
            normalize (bool): If True, the tensor is normalized. (Default to False)
            global_step(Optional[int]): current step.
            batch_idx (Optional[int]): current batch index to visualize. If None,
                visualize the entire batch.
            indexing_dict (Optional[dict]): a dictionary of the {layer_name: indexing}.
                where indexing is numpy-like fancy indexing.
            heatmap (bool): whether to add heatmap to the weights/ activations.
        """
        for name, array in weight_activation_dict.items():
            if batch_idx is None:
                # Select all items in the batch if batch_idx is not provided.
                batch_idx = list(range(array.shape[0]))
            if indexing_dict is not None:
                fancy_indexing = indexing_dict[name]
                fancy_indexing = (batch_idx,) + fancy_indexing
                array = array[fancy_indexing]
            else:
                array = array[batch_idx]
            add_ndim_array(
                self.writer,
                array,
                tag + name,
                normalize=normalize,
                global_step=global_step,
                heat_map=heat_map,
            )

    def flush(self):
        self.writer.flush()

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
            sub_cmtx,
            num_classes=len(subset_ids),
            class_names=sub_names,
            figsize=figsize,
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
        global_step (Optional[int]): current step.
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
                class_names[i],
                torch.Tensor(pred),
                k,
                class_names,
                figsize=figsize,
            )
            writer.add_figure(
                tag="Top {} predictions by classes/{}".format(
                    k, class_names[i]
                ),
                figure=hist,
                global_step=global_step,
            )


def add_ndim_array(
    writer,
    array,
    name,
    nrow=None,
    normalize=False,
    global_step=None,
    heat_map=True,
):
    """
    Visualize and add tensors of n-dimentionals to a Tensorboard SummaryWriter. Tensors
    will be visualized as a 2D grid image.
    Args:
        writer (SummaryWriter): Tensorboard SummaryWriter.
        array (tensor): tensor to visualize.
        name (str): name of the tensor.
        nrow (Optional[int]): number of 2D filters in each row in the grid image.
        normalize (bool): whether to normalize when we have multiple 2D filters.
            Default to False.
        global_step (Optional[int]): current step.
        heat_map (bool): whether to add heat map to 2D each 2D filters in array.
    """
    if array is not None and array.ndim != 0:
        if array.ndim == 1:
            reshaped_array = array.unsqueeze(0)
            if nrow is None:
                nrow = int(math.sqrt(reshaped_array.size()[1]))
            reshaped_array = reshaped_array.view(-1, nrow)
            if heat_map:
                reshaped_array = add_heatmap(reshaped_array)
                writer.add_image(
                    name,
                    reshaped_array,
                    global_step=global_step,
                    dataformats="CHW",
                )
            else:
                writer.add_image(
                    name,
                    reshaped_array,
                    global_step=global_step,
                    dataformats="HW",
                )
        elif array.ndim == 2:
            reshaped_array = array
            if heat_map:
                heatmap = add_heatmap(reshaped_array)
                writer.add_image(
                    name, heatmap, global_step=global_step, dataformats="CHW"
                )
            else:
                writer.add_image(
                    name,
                    reshaped_array,
                    global_step=global_step,
                    dataformats="HW",
                )
        else:
            last2_dims = array.size()[-2:]
            reshaped_array = array.view(-1, *last2_dims)
            if heat_map:
                reshaped_array = [
                    add_heatmap(array_2d).unsqueeze(0)
                    for array_2d in reshaped_array
                ]
                reshaped_array = torch.cat(reshaped_array, dim=0)
            else:
                reshaped_array = reshaped_array.unsqueeze(1)
            if nrow is None:
                nrow = int(math.sqrt(reshaped_array.size()[0]))
            img_grid = make_grid(
                reshaped_array, nrow, padding=1, normalize=normalize
            )
            writer.add_image(name, img_grid, global_step=global_step)


def add_heatmap(tensor):
    """
    Add heatmap to 2D tensor.
    Args:
        tensor (tensor): a 2D tensor. Tensor value must be in [0..1] range.
    Returns:
        heatmap (tensor): a 3D tensor. Result of applying heatmap to the 2D tensor.
    """
    assert tensor.ndim == 2, "Only support 2D tensors."
    # Move tensor to cpu if necessary.
    if tensor.device != torch.device("cpu"):
        arr = tensor.cpu()
    else:
        arr = tensor
    arr = arr.numpy()
    # Get the color map by name.
    cm = plt.get_cmap("viridis")
    heatmap = cm(arr)
    heatmap = heatmap[:, :, :3]
    # Convert (H, W, C) to (C, H, W)
    heatmap = torch.Tensor(heatmap).permute(2, 0, 1)
    return heatmap
