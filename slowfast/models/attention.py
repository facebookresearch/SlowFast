#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.


import torch
import torch.nn as nn
from collections import OrderedDict
from slowfast.models.common import Permute


def attention_pool(tensor, pool, thw_shape, has_cls_embed=True):
    assert tensor.ndim == 4
    if has_cls_embed:
        cls_tok, tensor = tensor[:, :, :1, :], tensor[:, :, 1:, :]

    B, N, L, C = tensor.shape
    T, H, W = thw_shape
    tensor = tensor.reshape(B * N, T, H, W, C).permute(0, 4, 1, 2, 3).contiguous()

    tensor = pool(tensor)

    thw_shape = [tensor.shape[2], tensor.shape[3], tensor.shape[4]]
    L_pooled = tensor.shape[2] * tensor.shape[3] * tensor.shape[4]
    tensor = tensor.reshape(B, N, C, L_pooled).transpose(2, 3)

    if has_cls_embed:
        tensor = torch.cat((cls_tok, tensor), dim=2)
    return tensor, thw_shape


class MultiScaleAttention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        proj_drop=0.0,
        kernel_q=(1, 1, 1),
        kernel_kv=(1, 1, 1),
        norm_layer=nn.LayerNorm,
        has_cls_embed=True,
        # Options include `conv`, `avg`, and `max`.
        mode="conv",
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.has_cls_embed = has_cls_embed
        stride_q = [int(q // 2) + 1 for q in kernel_q]
        stride_kv = [int(kv // 2) + 1 for kv in kernel_kv]
        padding_q = [int(q // 2) for q in kernel_q]
        padding_kv = [int(kv // 2) for kv in kernel_kv]

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.pool_q = None
        self.pool_kv = kernel_kv != (1, 1, 1)

        if mode == "avg":
            self.pool_q = nn.AvgPool3d(kernel_q, stride_q, padding_q, ceil_mode=False)
            self.pool_k = nn.AvgPool3d(kernel_kv, stride_kv, padding_kv, ceil_mode=False)
            self.pool_v = nn.AvgPool3d(kernel_kv, stride_kv, padding_kv, ceil_mode=False)
        elif mode == "max":
            self.pool_q = nn.MaxPool3d(kernel_q, stride_q, padding_q, ceil_mode=False)
            self.pool_k = nn.MaxPool3d(kernel_kv, stride_kv, padding_kv, ceil_mode=False)
            self.pool_v = nn.MaxPool3d(kernel_kv, stride_kv, padding_kv, ceil_mode=False)
        elif mode == "conv":
            self.pool_q = nn.Sequential(OrderedDict({
                'conv': nn.Conv3d(
                    head_dim,
                    head_dim,
                    kernel_q,
                    stride=stride_q,
                    padding=padding_q,
                    groups=head_dim,
                    bias=False,
                ),
                "permute_a": Permute((0, 2, 3, 4, 1)),  # BCTHW -> BTHWC
                'norm': norm_layer(head_dim),
                "permute_b": Permute((0, 4, 1, 2, 3)),  # BTHWC -> BCTHW
            }))
            self.pool_k = nn.Sequential(OrderedDict({
                'conv': nn.Conv3d(
                    head_dim,
                    head_dim,
                    kernel_kv,
                    stride=stride_kv,
                    padding=padding_kv,
                    groups=head_dim,
                    bias=False,
                ),
                "permute_a": Permute((0, 2, 3, 4, 1)),  # BCTHW -> BTHWC
                'norm': norm_layer(head_dim),
                "permute_b": Permute((0, 4, 1, 2, 3)),  # BTHWC -> BCTHW
            }))
            self.pool_v = nn.Sequential(OrderedDict({
                'conv': nn.Conv3d(
                    head_dim,
                    head_dim,
                    kernel_kv,
                    stride=stride_kv,
                    padding=padding_kv,
                    groups=head_dim,
                    bias=False,
                ),
                "permute_a": Permute((0, 2, 3, 4, 1)),  # BCTHW -> BTHWC
                'norm': norm_layer(head_dim),
                "permute_b": Permute((0, 4, 1, 2, 3)),  # BTHWC -> BCTHW
            }))
        else:
            raise NotImplementedError(f"Unsupported model {mode}")

    def forward(self, x, thw_shape):
        B, N, C = x.shape

        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        if self.pool_q:
            q, out_shape = attention_pool(q, self.pool_q, thw_shape, has_cls_embed=self.has_cls_embed)
        if self.pool_k:
            k, _ = attention_pool(k, self.pool_k, thw_shape, has_cls_embed=self.has_cls_embed)
        if self.pool_v:
            v, _ = attention_pool(v, self.pool_v, thw_shape, has_cls_embed=self.has_cls_embed)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        N = q.shape[2]
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, out_shape
