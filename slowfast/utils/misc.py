#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import json
import logging
import math
import numpy as np
import os
from datetime import datetime
import psutil
import torch
import torchvision.io as io
from fvcore.nn.activation_count import activation_count
from fvcore.nn.flop_count import flop_count
from matplotlib import pyplot as plt
from torch import nn
from torchvision.utils import make_grid

import slowfast.utils.logging as logging
import slowfast.utils.multiprocessing as mpu
from slowfast.datasets.utils import pack_pathway_output
from slowfast.models.batchnorm_helper import SubBatchNorm3d
from slowfast.utils.env import pathmgr

logger = logging.get_logger(__name__)


def check_nan_losses(loss):
    """
    Determine whether the loss is NaN (not a number).
    Args:
        loss (loss): loss to check whether is NaN.
    """
    if math.isnan(loss):
        raise RuntimeError("ERROR: Got NaN losses {}".format(datetime.now()))


def params_count(model, ignore_bn=False):
    """
    Compute the number of parameters.
    Args:
        model (model): model to count the number of parameters.
    """
    if not ignore_bn:
        return np.sum([p.numel() for p in model.parameters()]).item()
    else:
        count = 0
        for m in model.modules():
            if not isinstance(m, nn.BatchNorm3d):
                for p in m.parameters(recurse=False):
                    count += p.numel()
    return count


def gpu_mem_usage():
    """
    Compute the GPU memory usage for the current device (GB).
    """
    if torch.cuda.is_available():
        mem_usage_bytes = torch.cuda.max_memory_allocated()
    else:
        mem_usage_bytes = 0
    return mem_usage_bytes / 1024**3


def cpu_mem_usage():
    """
    Compute the system memory (RAM) usage for the current device (GB).
    Returns:
        usage (float): used memory (GB).
        total (float): total memory (GB).
    """
    vram = psutil.virtual_memory()
    usage = (vram.total - vram.available) / 1024**3
    total = vram.total / 1024**3

    return usage, total


def _get_model_analysis_input(cfg, use_train_input):
    """
    Return a dummy input for model analysis with batch size 1. The input is
        used for analyzing the model (counting flops and activations etc.).
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        use_train_input (bool): if True, return the input for training. Otherwise,
            return the input for testing.

    Returns:
        inputs: the input for model analysis.
    """
    rgb_dimension = 3
    if use_train_input:
        if "imagenet" in cfg.TRAIN.DATASET:
            input_tensors = torch.rand(
                rgb_dimension,
                cfg.DATA.TRAIN_CROP_SIZE,
                cfg.DATA.TRAIN_CROP_SIZE,
            )
        else:
            input_tensors = torch.rand(
                rgb_dimension,
                cfg.DATA.NUM_FRAMES,
                cfg.DATA.TRAIN_CROP_SIZE,
                cfg.DATA.TRAIN_CROP_SIZE,
            )
    else:
        if "imagenet" in cfg.TEST.DATASET:
            input_tensors = torch.rand(
                rgb_dimension,
                cfg.DATA.TEST_CROP_SIZE,
                cfg.DATA.TEST_CROP_SIZE,
            )
        else:
            input_tensors = torch.rand(
                rgb_dimension,
                cfg.DATA.NUM_FRAMES,
                cfg.DATA.TEST_CROP_SIZE,
                cfg.DATA.TEST_CROP_SIZE,
            )
    model_inputs = pack_pathway_output(cfg, input_tensors)
    for i in range(len(model_inputs)):
        model_inputs[i] = model_inputs[i].unsqueeze(0)
        if cfg.NUM_GPUS:
            model_inputs[i] = model_inputs[i].cuda(non_blocking=True)

    # If detection is enabled, count flops for one proposal.
    if cfg.DETECTION.ENABLE:
        bbox = torch.tensor([[0, 0, 1.0, 0, 1.0]])
        if cfg.NUM_GPUS:
            bbox = bbox.cuda()
        inputs = (model_inputs, bbox)
    else:
        inputs = (model_inputs,)
    return inputs


def get_model_stats(model, cfg, mode, use_train_input):
    """
    Compute statistics for the current model given the config.
    Args:
        model (model): model to perform analysis.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        mode (str): Options include `flop` or `activation`. Compute either flop
            (gflops) or activation count (mega).
        use_train_input (bool): if True, compute statistics for training. Otherwise,
            compute statistics for testing.

    Returns:
        float: the total number of count of the given model.
    """
    assert mode in [
        "flop",
        "activation",
    ], "'{}' not supported for model analysis".format(mode)
    if mode == "flop":
        model_stats_fun = flop_count
    elif mode == "activation":
        model_stats_fun = activation_count

    # Set model to evaluation mode for analysis.
    # Evaluation mode can avoid getting stuck with sync batchnorm.
    model_mode = model.training
    model.eval()
    inputs = _get_model_analysis_input(cfg, use_train_input)
    count_dict, *_ = model_stats_fun(model, inputs)
    count = sum(count_dict.values())
    model.train(model_mode)
    return count


def log_model_info(model, cfg, use_train_input=True):
    """
    Log info, includes number of parameters, gpu usage, gflops and activation count.
        The model info is computed when the model is in validation mode.
    Args:
        model (model): model to log the info.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        use_train_input (bool): if True, log info for training. Otherwise,
            log info for testing.
    """
    logger.info("Model:\n{}".format(model))
    params = params_count(model)
    logger.info("Params: {:,}".format(params))
    logger.info("Mem: {:,} MB".format(gpu_mem_usage()))
    flops = get_model_stats(model, cfg, "flop", use_train_input)
    logger.info("Flops: {:,} G".format(flops))
    logger.info(
        "Activations: {:,} M".format(
            get_model_stats(model, cfg, "activation", use_train_input)
        )
    )
    logger.info("nvidia-smi")
    os.system("nvidia-smi")
    return flops, params


def is_eval_epoch(cfg, cur_epoch, multigrid_schedule):
    """
    Determine if the model should be evaluated at the current epoch.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        cur_epoch (int): current epoch.
        multigrid_schedule (List): schedule for multigrid training.
    """
    if cur_epoch + 1 == cfg.SOLVER.MAX_EPOCH:
        return True
    if multigrid_schedule is not None:
        prev_epoch = 0
        for s in multigrid_schedule:
            if cur_epoch < s[-1]:
                period = max(
                    (s[-1] - prev_epoch) // cfg.MULTIGRID.EVAL_FREQ + 1, 1
                )
                return (s[-1] - 1 - cur_epoch) % period == 0
            prev_epoch = s[-1]

    return (cur_epoch + 1) % cfg.TRAIN.EVAL_PERIOD == 0


def plot_input(tensor, bboxes=(), texts=(), path="./tmp_vis.png"):
    """
    Plot the input tensor with the optional bounding box and save it to disk.
    Args:
        tensor (tensor): a tensor with shape of `NxCxHxW`.
        bboxes (tuple): bounding boxes with format of [[x, y, h, w]].
        texts (tuple): a tuple of string to plot.
        path (str): path to the image to save to.
    """
    tensor = tensor.float()
    tensor = tensor - tensor.min()
    tensor = tensor / tensor.max()
    f, ax = plt.subplots(nrows=1, ncols=tensor.shape[0], figsize=(50, 20))
    for i in range(tensor.shape[0]):
        ax[i].axis("off")
        ax[i].imshow(tensor[i].permute(1, 2, 0))
        # ax[1][0].axis('off')
        if bboxes is not None and len(bboxes) > i:
            for box in bboxes[i]:
                x1, y1, x2, y2 = box
                ax[i].vlines(x1, y1, y2, colors="g", linestyles="solid")
                ax[i].vlines(x2, y1, y2, colors="g", linestyles="solid")
                ax[i].hlines(y1, x1, x2, colors="g", linestyles="solid")
                ax[i].hlines(y2, x1, x2, colors="g", linestyles="solid")

        if texts is not None and len(texts) > i:
            ax[i].text(0, 0, texts[i])
    f.savefig(path)


def plot_input_normed(
    tensor,
    bboxes=(),
    texts=(),
    path="./tmp_vis.png",
    folder_path="",
    make_grids=False,
    output_video=False,
):
    """
    Plot the input tensor with the optional bounding box and save it to disk.
    Args:
        tensor (tensor): a tensor with shape of `NxCxHxW`.
        bboxes (tuple): bounding boxes with format of [[x, y, h, w]].
        texts (tuple): a tuple of string to plot.
        path (str): path to the image to save to.
    """
    tensor = tensor.float()
    try:
        os.mkdir(folder_path)
    except Exception as e:
        pass
    tensor = convert_normalized_images(tensor)
    if output_video:
        # assert make_grids, "video needs to have make_grids on"
        assert tensor.ndim == 5
        sz = tensor.shape

        if make_grids:
            vid = tensor.reshape([sz[0], sz[1] * sz[2], sz[3], sz[4]])
            vid = make_grid(vid, padding=8, pad_value=1.0, nrow=sz[0])
            vid = vid.reshape([sz[1], sz[2], vid.shape[1], vid.shape[2]])
        else:
            vid = tensor.reshape([sz[0] * sz[1], sz[2], sz[3], sz[4]])

        vid = vid.permute([0, 2, 3, 1])
        vid *= 255.0
        vid = vid.to(torch.uint8)
        fps = 30.0 * vid.shape[0] / 64.0
        io.video.write_video(path, vid, fps, video_codec="libx264")
    elif make_grids:
        if tensor.ndim > 4 and tensor.shape[0] == 1:
            tensor = tensor.squeeze()
            nrow = 1
        elif tensor.ndim == 5:
            nrow = tensor.shape[1]
            tensor = tensor.reshape(
                shape=(-1, tensor.shape[2], tensor.shape[3], tensor.shape[4])
            )
        vis2 = (
            make_grid(tensor, padding=8, pad_value=1.0, nrow=nrow)
            .permute(1, 2, 0)
            .cpu()
            .numpy()
        )
        plt.imsave(fname=path, arr=vis2, format="png")
    else:
        f, ax = plt.subplots(
            nrows=tensor.shape[0],
            ncols=tensor.shape[1],
            figsize=(10 * tensor.shape[1], 10 * tensor.shape[0]),
        )

        if tensor.shape[0] == 1:
            for i in range(tensor.shape[1]):
                ax[i].axis("off")
                ax[i].imshow(tensor[0][i].permute(1, 2, 0))
                # ax[1][0].axis('off')
                if bboxes is not None and len(bboxes) > i:
                    for box in bboxes[i]:
                        x1, y1, x2, y2 = box
                        ax[i].vlines(x1, y1, y2, colors="g", linestyles="solid")
                        ax[i].vlines(x2, y1, y2, colors="g", linestyles="solid")
                        ax[i].hlines(y1, x1, x2, colors="g", linestyles="solid")
                        ax[i].hlines(y2, x1, x2, colors="g", linestyles="solid")

            if texts is not None and len(texts) > i:
                ax[i].text(0, 0, texts[i])
        else:
            for i in range(tensor.shape[0]):
                for j in range(tensor.shape[1]):
                    ax[i][j].axis("off")
                    ax[i][j].imshow(tensor[i][j].permute(1, 2, 0))
                    # ax[1][0].axis('off')
                    if bboxes is not None and len(bboxes) > i:
                        for box in bboxes[i]:
                            x1, y1, x2, y2 = box
                            ax[i].vlines(
                                x1, y1, y2, colors="g", linestyles="solid"
                            )
                            ax[i].vlines(
                                x2, y1, y2, colors="g", linestyles="solid"
                            )
                            ax[i].hlines(
                                y1, x1, x2, colors="g", linestyles="solid"
                            )
                            ax[i].hlines(
                                y2, x1, x2, colors="g", linestyles="solid"
                            )

                    if texts is not None and len(texts) > i:
                        ax[i].text(0, 0, texts[i])
        print(f"{path}")
        f.tight_layout(pad=0.0)
        with pathmgr.open(path, "wb") as h:
            f.savefig(h)


def convert_normalized_images(tensor):

    tensor = tensor * 0.225
    tensor = tensor + 0.45

    tensor = tensor.clamp(min=0.0, max=1.0)

    return tensor


def frozen_bn_stats(model):
    """
    Set all the bn layers to eval mode.
    Args:
        model (model): model to set bn layers to eval mode.
    """
    for m in model.modules():
        if isinstance(m, nn.BatchNorm3d):
            m.eval()


def aggregate_sub_bn_stats(module):
    """
    Recursively find all SubBN modules and aggregate sub-BN stats.
    Args:
        module (nn.Module)
    Returns:
        count (int): number of SubBN module found.
    """
    count = 0
    for child in module.children():
        if isinstance(child, SubBatchNorm3d):
            child.aggregate_stats()
            count += 1
        else:
            count += aggregate_sub_bn_stats(child)
    return count


def launch_job(cfg, init_method, func, daemon=False):
    """
    Run 'func' on one or more GPUs, specified in cfg
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        init_method (str): initialization method to launch the job with multiple
            devices.
        func (function): job to run on GPU(s)
        daemon (bool): The spawned processes’ daemon flag. If set to True,
            daemonic processes will be created
    """
    if cfg.NUM_GPUS > 1:
        torch.multiprocessing.spawn(
            mpu.run,
            nprocs=cfg.NUM_GPUS,
            args=(
                cfg.NUM_GPUS,
                func,
                init_method,
                cfg.SHARD_ID,
                cfg.NUM_SHARDS,
                cfg.DIST_BACKEND,
                cfg,
            ),
            daemon=daemon,
        )
    else:
        func(cfg=cfg)


def get_class_names(path, parent_path=None, subset_path=None):
    """
    Read json file with entries {classname: index} and return
    an array of class names in order.
    If parent_path is provided, load and map all children to their ids.
    Args:
        path (str): path to class ids json file.
            File must be in the format {"class1": id1, "class2": id2, ...}
        parent_path (Optional[str]): path to parent-child json file.
            File must be in the format {"parent1": ["child1", "child2", ...], ...}
        subset_path (Optional[str]): path to text file containing a subset
            of class names, separated by newline characters.
    Returns:
        class_names (list of strs): list of class names.
        class_parents (dict): a dictionary where key is the name of the parent class
            and value is a list of ids of the children classes.
        subset_ids (list of ints): list of ids of the classes provided in the
            subset file.
    """
    try:
        with pathmgr.open(path, "r") as f:
            class2idx = json.load(f)
    except Exception as err:
        print("Fail to load file from {} with error {}".format(path, err))
        return

    max_key = max(class2idx.values())
    class_names = [None] * (max_key + 1)

    for k, i in class2idx.items():
        class_names[i] = k

    class_parent = None
    if parent_path is not None and parent_path != "":
        try:
            with pathmgr.open(parent_path, "r") as f:
                d_parent = json.load(f)
        except EnvironmentError as err:
            print(
                "Fail to load file from {} with error {}".format(
                    parent_path, err
                )
            )
            return
        class_parent = {}
        for parent, children in d_parent.items():
            indices = [
                class2idx[c] for c in children if class2idx.get(c) is not None
            ]
            class_parent[parent] = indices

    subset_ids = None
    if subset_path is not None and subset_path != "":
        try:
            with pathmgr.open(subset_path, "r") as f:
                subset = f.read().split("\n")
                subset_ids = [
                    class2idx[name]
                    for name in subset
                    if class2idx.get(name) is not None
                ]
        except EnvironmentError as err:
            print(
                "Fail to load file from {} with error {}".format(
                    subset_path, err
                )
            )
            return

    return class_names, class_parent, subset_ids
