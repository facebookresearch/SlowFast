#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Functions that handle saving and loading of checkpoints."""

import os
import pickle
from collections import OrderedDict
import torch

import slowfast.utils.distributed as du
import slowfast.utils.logging as logging
from slowfast.utils.c2_model_loading import get_name_convert_func

logger = logging.get_logger(__name__)


def make_checkpoint_dir(path_to_job):
    """
    Creates the checkpoint directory (if not present already).
    Args:
        path_to_job (string): the path to the folder of the current job.
    """
    checkpoint_dir = os.path.join(path_to_job, "checkpoints")
    # Create the checkpoint dir from the master process
    if du.is_master_proc() and not os.path.exists(checkpoint_dir):
        try:
            os.makedirs(checkpoint_dir)
        except Exception:
            pass
    return checkpoint_dir


def get_checkpoint_dir(path_to_job):
    """
    Get path for storing checkpoints.
    Args:
        path_to_job (string): the path to the folder of the current job.
    """
    return os.path.join(path_to_job, "checkpoints")


def get_path_to_checkpoint(path_to_job, epoch):
    """
    Get the full path to a checkpoint file.
    Args:
        path_to_job (string): the path to the folder of the current job.
        epoch (int): the number of epoch for the checkpoint.
    """
    name = "checkpoint_epoch_{:05d}.pyth".format(epoch)
    return os.path.join(get_checkpoint_dir(path_to_job), name)


def get_last_checkpoint(path_to_job):
    """
    Get the last checkpoint from the checkpointing folder.
    Args:
        path_to_job (string): the path to the folder of the current job.
    """

    d = get_checkpoint_dir(path_to_job)
    names = os.listdir(d) if os.path.exists(d) else []
    names = [f for f in names if "checkpoint" in f]
    assert len(names), "No checkpoints found in '{}'.".format(d)
    # Sort the checkpoints by epoch.
    name = sorted(names)[-1]
    return os.path.join(d, name)


def has_checkpoint(path_to_job):
    """
    Determines if the given directory contains a checkpoint.
    Args:
        path_to_job (string): the path to the folder of the current job.
    """
    d = get_checkpoint_dir(path_to_job)
    files = os.listdir(d) if os.path.exists(d) else []
    return any("checkpoint" in f for f in files)


def is_checkpoint_epoch(cur_epoch, checkpoint_period):
    """
    Determine if a checkpoint should be saved on current epoch.
    Args:
        cur_epoch (int): current number of epoch of the model.
        checkpoint_period (int): the frequency of checkpointing.
    """
    return (cur_epoch + 1) % checkpoint_period == 0


def save_checkpoint(path_to_job, model, optimizer, epoch, cfg):
    """
    Save a checkpoint.
    Args:
        model (model): model to save the weight to the checkpoint.
        optimizer (optim): optimizer to save the historical state.
        epoch (int): current number of epoch of the model.
        cfg (CfgNode): configs to save.
    """
    # Save checkpoints only from the master process.
    if not du.is_master_proc(cfg.NUM_GPUS * cfg.NUM_SHARDS):
        return
    # Ensure that the checkpoint dir exists.
    os.makedirs(get_checkpoint_dir(path_to_job), exist_ok=True)
    # Omit the DDP wrapper in the multi-gpu setting.
    sd = model.module.state_dict() if cfg.NUM_GPUS > 1 else model.state_dict()
    # Record the state.
    checkpoint = {
        "epoch": epoch,
        "model_state": sd,
        "optimizer_state": optimizer.state_dict(),
        "cfg": cfg.dump(),
    }
    # Write the checkpoint.
    path_to_checkpoint = get_path_to_checkpoint(path_to_job, epoch + 1)
    torch.save(checkpoint, path_to_checkpoint)
    return path_to_checkpoint


def inflate_weight(state_dict_2d, state_dict_3d):
    """
    Inflate 2D model weights in state_dict_2d to the 3D model weights in
    state_dict_3d. The details can be found in:
    Joao Carreira, and Andrew Zisserman.
    "Quo vadis, action recognition? a new model and the kinetics dataset."
    Args:
        state_dict_2d (OrderedDict): a dict of parameters from a 2D model.
        state_dict_3d (OrderedDict): a dict of parameters from a 3D model.
    Returns:
        state_dict_inflated (OrderedDict): a dict of inflated parameters.
    """
    state_dict_inflated = OrderedDict()
    for k, v2d in state_dict_2d.items():
        assert k in state_dict_3d.keys()
        v3d = state_dict_3d[k]
        # Inflate the weight of 2D conv to 3D conv.
        if len(v2d.shape) == 4 and len(v3d.shape) == 5:
            logger.info(
                "Inflate {}: {} -> {}: {}".format(k, v2d.shape, k, v3d.shape)
            )
            # Dimension need to be match.
            assert v2d.shape[-2:] == v3d.shape[-2:]
            assert v2d.shape[:2] == v3d.shape[:2]
            v3d = (
                v2d.unsqueeze(2).repeat(1, 1, v3d.shape[2], 1, 1) / v3d.shape[2]
            )
        if v2d.shape == v3d.shape:
            v3d = v2d
        state_dict_inflated[k] = v3d.clone()
    return state_dict_inflated


def load_checkpoint(
    path_to_checkpoint,
    model,
    data_parallel=True,
    optimizer=None,
    inflation=False,
    convert_from_caffe2=False,
):
    """
    Load the checkpoint from the given file. If inflation is True, inflate the
    2D Conv weights from the checkpoint to 3D Conv.
    Args:
        path_to_checkpoint (string): path to the checkpoint to load.
        model (model): model to load the weights from the checkpoint.
        data_parallel (bool): if true, model is wrapped by
        torch.nn.parallel.DistributedDataParallel.
        optimizer (optim): optimizer to load the historical state.
        inflation (bool): if True, inflate the weights from the checkpoint.
        convert_from_caffe2 (bool): if True, load the model from caffe2 and
            convert it to pytorch.
    Returns:
        (int): the number of training epoch of the checkpoint.
    """
    assert os.path.exists(
        path_to_checkpoint
    ), "Checkpoint '{}' not found".format(path_to_checkpoint)
    # Account for the DDP wrapper in the multi-gpu setting.
    ms = model.module if data_parallel else model
    if convert_from_caffe2:
        with open(path_to_checkpoint, "rb") as f:
            caffe2_checkpoint = pickle.load(f, encoding="latin1")
        state_dict = OrderedDict()
        name_convert_func = get_name_convert_func()
        for key in caffe2_checkpoint["blobs"].keys():
            converted_key = name_convert_func(key)
            if converted_key in ms.state_dict():
                if caffe2_checkpoint["blobs"][key].shape == tuple(
                    ms.state_dict()[converted_key].shape
                ):
                    state_dict[converted_key] = torch.tensor(
                        caffe2_checkpoint["blobs"][key]
                    ).clone()
                    logger.info(
                        "{}: {} => {}: {}".format(
                            key,
                            caffe2_checkpoint["blobs"][key].shape,
                            converted_key,
                            tuple(ms.state_dict()[converted_key].shape),
                        )
                    )
                else:
                    logger.warn(
                        "!! {}: {} does not match {}: {}".format(
                            key,
                            caffe2_checkpoint["blobs"][key].shape,
                            converted_key,
                            tuple(ms.state_dict()[converted_key].shape),
                        )
                    )
            else:
                if not any(
                    prefix in key for prefix in ["momentum", "lr", "model_iter"]
                ):
                    logger.warn(
                        "!! {}: can not be converted, got {}".format(
                            key, converted_key
                        )
                    )
        ms.load_state_dict(state_dict, strict=False)
        epoch = -1
    else:
        # Load the checkpoint on CPU to avoid GPU mem spike.
        checkpoint = torch.load(path_to_checkpoint, map_location="cpu")
        if inflation:
            # Try to inflate the model.
            model_state_dict_3d = (
                model.module.state_dict()
                if data_parallel
                else model.state_dict()
            )
            inflated_model_dict = inflate_weight(
                checkpoint["model_state"], model_state_dict_3d
            )
            ms.load_state_dict(inflated_model_dict, strict=False)
        else:
            ms.load_state_dict(checkpoint["model_state"])
            # Load the optimizer state (commonly not done when fine-tuning)
            if optimizer:
                optimizer.load_state_dict(checkpoint["optimizer_state"])
        if "epoch" in checkpoint.keys():
            epoch = checkpoint["epoch"]
        else:
            epoch = -1
    return epoch
