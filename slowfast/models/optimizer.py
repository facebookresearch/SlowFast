#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Optimizer."""

import torch

import slowfast.utils.lr_policy as lr_policy


def construct_optimizer(model, cfg):
    """
    Construct a stochastic gradient descent or ADAM optimizer with momentum.
    Details can be found in:
    Herbert Robbins, and Sutton Monro. "A stochastic approximation method."
    and
    Diederik P.Kingma, and Jimmy Ba.
    "Adam: A Method for Stochastic Optimization."

    Args:
        model (model): model to perform stochastic gradient descent
        optimization or ADAM optimization.
        cfg (config): configs of hyper-parameters of SGD or ADAM, includes base
        learning rate,  momentum, weight_decay, dampening, and etc.
    """
    if cfg.SOLVER.LAYER_DECAY > 0.0 and cfg.SOLVER.LAYER_DECAY < 1.0:
        optim_params = get_param_groups(model, cfg)
    elif cfg.SOLVER.LAYER_DECAY == 1.0:
        bn_parameters = []
        non_bn_parameters = []
        zero_parameters = []
        no_grad_parameters = []
        skip = {}

        if cfg.NUM_GPUS > 1:
            if hasattr(model.module, "no_weight_decay"):
                skip = model.module.no_weight_decay()
        else:
            if hasattr(model, "no_weight_decay"):
                skip = model.no_weight_decay()

        for name_m, m in model.named_modules():
            is_bn = isinstance(m, torch.nn.modules.batchnorm._NormBase)
            for name_p, p in m.named_parameters(recurse=False):
                name = "{}.{}".format(name_m, name_p).strip(".")
                if not p.requires_grad:
                    no_grad_parameters.append(p)
                elif is_bn:
                    bn_parameters.append(p)
                elif any(k in name for k in skip):
                    zero_parameters.append(p)
                elif cfg.SOLVER.ZERO_WD_1D_PARAM and (
                    len(p.shape) == 1 or name.endswith(".bias")
                ):
                    zero_parameters.append(p)
                else:
                    non_bn_parameters.append(p)

        optim_params = [
            {
                "params": bn_parameters,
                "weight_decay": cfg.BN.WEIGHT_DECAY,
                "layer_decay": 1.0,
                "apply_LARS": False,
            },
            {
                "params": non_bn_parameters,
                "weight_decay": cfg.SOLVER.WEIGHT_DECAY,
                "layer_decay": 1.0,
                "apply_LARS": cfg.SOLVER.LARS_ON,
            },
            {
                "params": zero_parameters,
                "weight_decay": 0.0,
                "layer_decay": 1.0,
                "apply_LARS": cfg.SOLVER.LARS_ON,
            },
        ]
        optim_params = [x for x in optim_params if len(x["params"])]

        # Check all parameters will be passed into optimizer.
        assert len(list(model.parameters())) == len(non_bn_parameters) + len(
            bn_parameters
        ) + len(zero_parameters) + len(
            no_grad_parameters
        ), "parameter size does not match: {} + {} + {} + {} != {}".format(
            len(non_bn_parameters),
            len(bn_parameters),
            len(zero_parameters),
            len(no_grad_parameters),
            len(list(model.parameters())),
        )
        print(
            "bn {}, non bn {}, zero {}, no grad {}".format(
                len(bn_parameters),
                len(non_bn_parameters),
                len(zero_parameters),
                len(no_grad_parameters),
            )
        )
    else:
        raise ValueError(
            "Layer decay should be in (0, 1], but is {}".format(
                cfg.SOLVER.LAYER_DECAY
            )
        )

    if cfg.SOLVER.OPTIMIZING_METHOD == "sgd":
        optimizer = torch.optim.SGD(
            optim_params,
            lr=cfg.SOLVER.BASE_LR,
            momentum=cfg.SOLVER.MOMENTUM,
            weight_decay=cfg.SOLVER.WEIGHT_DECAY,
            dampening=cfg.SOLVER.DAMPENING,
            nesterov=cfg.SOLVER.NESTEROV,
        )
    elif cfg.SOLVER.OPTIMIZING_METHOD == "adam":
        optimizer = torch.optim.Adam(
            optim_params,
            lr=cfg.SOLVER.BASE_LR,
            betas=cfg.SOLVER.BETAS,
            weight_decay=cfg.SOLVER.WEIGHT_DECAY,
        )
    elif cfg.SOLVER.OPTIMIZING_METHOD == "adamw":
        optimizer = torch.optim.AdamW(
            optim_params,
            lr=cfg.SOLVER.BASE_LR,
            betas=cfg.SOLVER.BETAS,
            eps=1e-08,
            weight_decay=cfg.SOLVER.WEIGHT_DECAY,
        )
    elif cfg.SOLVER.OPTIMIZING_METHOD == "mt_adamw":
        optimizer = torch.optim._multi_tensor.AdamW(
            optim_params,
            lr=cfg.SOLVER.BASE_LR,
            betas=cfg.SOLVER.BETAS,
            eps=1e-08,
            weight_decay=cfg.SOLVER.WEIGHT_DECAY,
        )
    else:
        raise NotImplementedError(
            "Does not support {} optimizer".format(cfg.SOLVER.OPTIMIZING_METHOD)
        )
    if cfg.SOLVER.LARS_ON:
        optimizer = LARS(
            optimizer=optimizer, trust_coefficient=0.001, clip=False
        )
    return optimizer


def get_param_groups(model, cfg):
    def _get_layer_decay(name):
        layer_id = None
        if name in ("cls_token", "mask_token"):
            layer_id = 0
        elif name.startswith("pos_embed"):
            layer_id = 0
        elif name.startswith("patch_embed"):
            layer_id = 0
        elif name.startswith("blocks"):
            layer_id = int(name.split(".")[1]) + 1
        else:
            layer_id = cfg.MVIT.DEPTH + 1
        layer_decay = cfg.SOLVER.LAYER_DECAY ** (cfg.MVIT.DEPTH + 1 - layer_id)
        return layer_id, layer_decay

    for m in model.modules():
        assert not isinstance(
            m, torch.nn.modules.batchnorm._NormBase
        ), "BN is not supported with layer decay"

    non_bn_parameters_count = 0
    zero_parameters_count = 0
    no_grad_parameters_count = 0
    parameter_group_names = {}
    parameter_group_vars = {}

    skip = {}
    if cfg.NUM_GPUS > 1:
        if hasattr(model.module, "no_weight_decay"):
            skip = model.module.no_weight_decay()
            # skip = {"module." + v for v in skip}
    else:
        if hasattr(model, "no_weight_decay"):
            skip = model.no_weight_decay()

    for name, p in model.named_parameters():
        if not p.requires_grad:
            group_name = "no_grad"
            no_grad_parameters_count += 1
            continue
        name = name[len("module.") :] if name.startswith("module.") else name
        if name in skip or (
            (len(p.shape) == 1 or name.endswith(".bias"))
            and cfg.SOLVER.ZERO_WD_1D_PARAM
        ):
            layer_id, layer_decay = _get_layer_decay(name)
            group_name = "layer_%d_%s" % (layer_id, "zero")
            weight_decay = 0.0
            zero_parameters_count += 1
        else:
            layer_id, layer_decay = _get_layer_decay(name)
            group_name = "layer_%d_%s" % (layer_id, "non_bn")
            weight_decay = cfg.SOLVER.WEIGHT_DECAY
            non_bn_parameters_count += 1

        if group_name not in parameter_group_names:
            parameter_group_names[group_name] = {
                "weight_decay": weight_decay,
                "params": [],
                "layer_decay": layer_decay,
            }
            parameter_group_vars[group_name] = {
                "weight_decay": weight_decay,
                "params": [],
                "layer_decay": layer_decay,
            }
        parameter_group_names[group_name]["params"].append(name)
        parameter_group_vars[group_name]["params"].append(p)

    # print("Param groups = %s" % json.dumps(parameter_group_names, indent=2))
    optim_params = list(parameter_group_vars.values())

    # Check all parameters will be passed into optimizer.
    assert (
        len(list(model.parameters()))
        == non_bn_parameters_count
        + zero_parameters_count
        + no_grad_parameters_count
    ), "parameter size does not match: {} + {} + {} != {}".format(
        non_bn_parameters_count,
        zero_parameters_count,
        no_grad_parameters_count,
        len(list(model.parameters())),
    )
    print(
        "non bn {}, zero {}, no grad {}".format(
            non_bn_parameters_count,
            zero_parameters_count,
            no_grad_parameters_count,
        )
    )

    return optim_params


def get_epoch_lr(cur_epoch, cfg):
    """
    Retrieves the lr for the given epoch (as specified by the lr policy).
    Args:
        cfg (config): configs of hyper-parameters of ADAM, includes base
        learning rate, betas, and weight decay.
        cur_epoch (float): the number of epoch of the current training stage.
    """
    return lr_policy.get_lr_at_epoch(cfg, cur_epoch)


def set_lr(optimizer, new_lr):
    """
    Sets the optimizer lr to the specified value.
    Args:
        optimizer (optim): the optimizer using to optimize the current network.
        new_lr (float): the new learning rate to set.
    """
    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lr * param_group["layer_decay"]


class LARS(object):
    """
    this class is adapted from https://github.com/NVIDIA/apex/blob/master/apex/parallel/LARC.py to
     include ignoring LARS application specific parameters (e.g. 1D params)

    Args:
        optimizer: Pytorch optimizer to wrap and modify learning rate for.
        trust_coefficient: Trust coefficient for calculating the lr. See https://arxiv.org/abs/1708.03888
        clip: Decides between clipping or scaling mode of LARS. If `clip=True` the learning rate is set to `min(optimizer_lr, local_lr)` for each parameter. If `clip=False` the learning rate is set to `local_lr*optimizer_lr`.
        eps: epsilon kludge to help with numerical stability while calculating adaptive_lr
    """

    def __init__(
        self,
        optimizer,
        trust_coefficient=0.02,
        clip=True,
        eps=1e-8,
        ignore_1d_param=True,
    ):
        self.optim = optimizer
        self.trust_coefficient = trust_coefficient
        self.eps = eps
        self.clip = clip
        self.ignore_1d_param = ignore_1d_param

    def __getstate__(self):
        return self.optim.__getstate__()

    def __setstate__(self, state):
        self.optim.__setstate__(state)

    @property
    def state(self):
        return self.optim.state

    def __repr__(self):
        return self.optim.__repr__()

    @property
    def param_groups(self):
        return self.optim.param_groups

    @param_groups.setter
    def param_groups(self, value):
        self.optim.param_groups = value

    def state_dict(self):
        return self.optim.state_dict()

    def load_state_dict(self, state_dict):
        self.optim.load_state_dict(state_dict)

    def zero_grad(self):
        self.optim.zero_grad()

    def add_param_group(self, param_group):
        self.optim.add_param_group(param_group)

    def step(self):
        with torch.no_grad():
            weight_decays = []
            for group in self.optim.param_groups:
                # absorb weight decay control from optimizer
                weight_decay = (
                    group["weight_decay"] if "weight_decay" in group else 0
                )
                weight_decays.append(weight_decay)
                apply_LARS = (
                    group["apply_LARS"] if "apply_LARS" in group else True
                )
                if not apply_LARS:
                    continue
                group["weight_decay"] = 0
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    if self.ignore_1d_param and p.ndim == 1:  # ignore bias
                        continue
                    param_norm = torch.norm(p.data)
                    grad_norm = torch.norm(p.grad.data)

                    if param_norm != 0 and grad_norm != 0:
                        # calculate adaptive lr + weight decay
                        adaptive_lr = (
                            self.trust_coefficient
                            * (param_norm)
                            / (grad_norm + param_norm * weight_decay + self.eps)
                        )

                        # clip learning rate for LARS
                        if self.clip:
                            # calculation of adaptive_lr so that when multiplied by lr it equals `min(adaptive_lr, lr)`
                            adaptive_lr = min(adaptive_lr / group["lr"], 1)

                        p.grad.data += weight_decay * p.data
                        p.grad.data *= adaptive_lr

        self.optim.step()
        # return weight decay control to optimizer
        for i, group in enumerate(self.optim.param_groups):
            group["weight_decay"] = weight_decays[i]


def get_grad_norm_(parameters, norm_type=2.0):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.0)
    device = parameters[0].grad.device
    if norm_type == "inf":
        total_norm = max(
            p.grad.detach().abs().max().to(device) for p in parameters
        )
    else:
        total_norm = torch.norm(
            torch.stack(
                [
                    torch.norm(p.grad.detach(), norm_type).to(device)
                    for p in parameters
                ]
            ),
            norm_type,
        )
    return total_norm
