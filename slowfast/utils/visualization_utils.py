#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import itertools
import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import confusion_matrix


def get_confusion_matrix(preds, labels, num_classes, normalize="true"):
    """
    Calculate confusion matrix on the provided preds and labels.
    Args:
        preds (tensor or lists of tensors): predictions. Each tensor is in
            in the shape of (n_batch, num_classes). Tensor(s) must be on CPU.
        labels (tensor or lists of tensors): corresponding labels. Each tensor is
            in the shape of either (n_batch,) or (n_batch, num_classes).
        num_classes (int): number of classes. Tensor(s) must be on CPU.
        normalize (Optional[str]) : {‘true’, ‘pred’, ‘all’}, default="true"
            Normalizes confusion matrix over the true (rows), predicted (columns)
            conditions or all the population. If None, confusion matrix
            will not be normalized.
    Returns:
        cmtx (ndarray): confusion matrix of size (num_classes x num_classes)
    """
    if isinstance(preds, list):
        preds = torch.cat(preds, dim=0)
    if isinstance(labels, list):
        labels = torch.cat(labels, dim=0)
    # If labels are one-hot encoded, get their indices.
    if labels.ndim == preds.ndim:
        labels = torch.argmax(labels, dim=-1)
    # Get the predicted class indices for examples.
    preds = torch.flatten(torch.argmax(preds, dim=-1))
    labels = torch.flatten(labels)
    cmtx = confusion_matrix(
        labels, preds, labels=list(range(num_classes)), normalize=normalize
    )
    return cmtx


def plot_confusion_matrix(cmtx, num_classes, class_names=None, figsize=None):
    """
    A function to create a colored and labeled confusion matrix matplotlib figure
    given true labels and preds.
    Args:
        cmtx (ndarray): confusion matrix.
        num_classes (int): total number of classes.
        class_names (Optional[list of strs]): a list of class names.
        figsize (Optional[float, float]): the figure size of the confusion matrix.
            If None, default to [6.4, 4.8].

    Returns:
        img (figure): matplotlib figure.
    """
    if class_names is None or type(class_names) != list:
        class_names = [str(i) for i in range(num_classes)]

    figure = plt.figure(figsize=figsize)
    plt.imshow(cmtx, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Use white text if squares are dark; otherwise black.
    threshold = cmtx.max() / 2.0
    for i, j in itertools.product(range(cmtx.shape[0]), range(cmtx.shape[1])):
        color = "white" if cmtx[i, j] > threshold else "black"
        plt.text(
            j,
            i,
            format(cmtx[i, j], ".2f") if cmtx[i, j] != 0 else ".",
            horizontalalignment="center",
            color=color,
        )

    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")

    return figure


def plot_topk_histogram(tag, array, k=10, class_names=None, figsize=None):
    """
    Plot histogram of top-k value from the given array.
    Args:
        tag (str): histogram title.
        array (tensor): a tensor to draw top k value from.
        k (int): number of top values to draw from array.
            Defaut to 10.
        class_names (list of strings, optional):
            a list of names for values in array.
        figsize (Optional[float, float]): the figure size of the confusion matrix.
            If None, default to [6.4, 4.8].
    Returns:
        fig (matplotlib figure): a matplotlib figure of the histogram.
    """
    val, ind = torch.topk(array, k)

    fig = plt.Figure(figsize=figsize, facecolor="w", edgecolor="k")

    ax = fig.add_subplot(1, 1, 1)

    if class_names is None:
        class_names = [str(i) for i in ind]
    else:
        class_names = [class_names[i] for i in ind]

    tick_marks = np.arange(k)
    width = 0.75
    ax.bar(
        tick_marks,
        val,
        width,
        color="orange",
        tick_label=class_names,
        edgecolor="w",
        linewidth=1,
    )

    ax.set_xlabel("Candidates")
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(class_names, rotation=-45, ha="center")
    ax.xaxis.set_label_position("bottom")
    ax.xaxis.tick_bottom()

    y_tick = np.linspace(0, 1, num=10)
    ax.set_ylabel("Frequency")
    ax.set_yticks(y_tick)
    y_labels = [format(i, ".1f") for i in y_tick]
    ax.set_yticklabels(y_labels, ha="center")

    for i, v in enumerate(val.numpy()):
        ax.text(
            i - 0.1,
            v + 0.03,
            format(v, ".2f"),
            color="orange",
            fontweight="bold",
        )

    ax.set_title(tag)

    fig.set_tight_layout(True)

    return fig
