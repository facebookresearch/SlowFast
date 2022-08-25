# Copyright (c) Facebook, Inc. and its affiliates.

import torch
import torch.nn as nn


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop_rate=0.0,
    ):
        super().__init__()
        self.drop_rate = drop_rate
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        if self.drop_rate > 0.0:
            self.drop = nn.Dropout(drop_rate)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        if self.drop_rate > 0.0:
            x = self.drop(x)
        x = self.fc2(x)
        if self.drop_rate > 0.0:
            x = self.drop(x)
        return x


class Permute(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.dims = dims

    def forward(self, x):
        return x.permute(*self.dims)


def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    """
    Stochastic Depth per sample.
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (
        x.ndim - 1
    )  # work with diff dim tensors, not just 2D ConvNets
    mask = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    mask.floor_()  # binarize
    output = x.div(keep_prob) * mask
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class TwoStreamFusion(nn.Module):
    def __init__(self, mode, dim=None, kernel=3, padding=1):
        """
        A general constructor for neural modules fusing two equal sized tensors
        in forward. Following options are supported:

        "add" / "max" / "min" / "avg"             : respective operations on the two halves.
        "concat"                                  : NOOP.
        "concat_linear_{dim_mult}_{drop_rate}"    : MLP to fuse with hidden dim "dim_mult"
                                                    (optional, def 1.) higher than input dim
                                                    with optional dropout "drop_rate" (def: 0.)
        "ln+concat_linear_{dim_mult}_{drop_rate}" : perform MLP after layernorm on the input.

        """
        super().__init__()
        self.mode = mode
        if mode == "add":
            self.fuse_fn = lambda x: torch.stack(torch.chunk(x, 2, dim=2)).sum(
                dim=0
            )
        elif mode == "max":
            self.fuse_fn = (
                lambda x: torch.stack(torch.chunk(x, 2, dim=2))
                .max(dim=0)
                .values
            )
        elif mode == "min":
            self.fuse_fn = (
                lambda x: torch.stack(torch.chunk(x, 2, dim=2))
                .min(dim=0)
                .values
            )
        elif mode == "avg":
            self.fuse_fn = lambda x: torch.stack(torch.chunk(x, 2, dim=2)).mean(
                dim=0
            )
        elif mode == "concat":
            # x itself is the channel concat version
            self.fuse_fn = lambda x: x
        elif "concat_linear" in mode:
            if len(mode.split("_")) == 2:
                dim_mult = 1.0
                drop_rate = 0.0
            elif len(mode.split("_")) == 3:
                dim_mult = float(mode.split("_")[-1])
                drop_rate = 0.0

            elif len(mode.split("_")) == 4:
                dim_mult = float(mode.split("_")[-2])
                drop_rate = float(mode.split("_")[-1])
            else:
                raise NotImplementedError

            if mode.split("+")[0] == "ln":
                self.fuse_fn = nn.Sequential(
                    nn.LayerNorm(dim),
                    Mlp(
                        in_features=dim,
                        hidden_features=int(dim * dim_mult),
                        act_layer=nn.GELU,
                        out_features=dim,
                        drop_rate=drop_rate,
                    ),
                )
            else:
                self.fuse_fn = Mlp(
                    in_features=dim,
                    hidden_features=int(dim * dim_mult),
                    act_layer=nn.GELU,
                    out_features=dim,
                    drop_rate=drop_rate,
                )

        else:
            raise NotImplementedError

    def forward(self, x):
        if "concat_linear" in self.mode:
            return self.fuse_fn(x) + x

        else:
            return self.fuse_fn(x)
