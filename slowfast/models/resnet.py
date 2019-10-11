#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""ResNet models."""

import pyvid.models.header_helper as head_helper
import pyvid.models.resnet_helper as resnet_helper
import pyvid.models.stem_helper as stem_helper
import pyvid.utils.logging as logging
import pyvid.utils.weight_init_helper as init_helper
import torch.nn as nn
from pyvid.core.config import cfg

logger = logging.get_logger(__name__)


# Stage depths for an ImageNet model {model depth -> (d2, d3, d4, d5)}
_MODEL_STAGE_DS = {50: (3, 4, 6, 3)}

# TODO: move this to configs
_MODEL_STAGE_TEMPORAL = {
    "c2d": (
        # temporal kenel sizes
        [
            [1],  # conv1 temporal kernel
            [1] * 3,  # res2 temporal kernel
            [1] * 4,  # res3 temporal kernel
            [1] * 6,  # res4 temporal kernel
            [1] * 3,  # res5 temporal kernel
        ],
        # temporal stride sizes
        [
            [1],  # conv1 temporal stride
            [1] * 3,  # res2 temporal stride
            [1] * 4,  # res3 temporal stride
            [1] * 6,  # res4 temporal stride
            [1] * 3,  # res5 temporal stride
        ],
        # temporal pool kernel sizes
        [2, 1],  # pool1 temporal kernel 1  # pool2 temporal kernel 2
    ),
    "i3d": (
        # temporal kenel sizes
        [
            [5],  # conv1 temporal kernel
            [3] * 3,  # res2 temporal kernel
            [3, 1] * 2,  # res3 temporal kernel
            [3, 1] * 3,  # res4 temporal kernel
            [1, 3, 1],  # res5 temporal kernel
        ],
        # temporal stride sizes
        [
            [1],  # conv1 temporal stride
            [1] * 3,  # res2 temporal stride
            [1] * 4,  # res3 temporal stride
            [1] * 6,  # res4 temporal stride
            [1] * 3,  # res5 temporal stride
        ],
        # temporal pool kernel sizes
        [2, 1],  # pool1 temporal kernel 1  # pool2 temporal kernel 2
    ),
    "slowOnly": (
        # temporal kenel sizes
        [
            [1],  # conv1 temporal kernel
            [1] * 3,  # res2 temporal kernel
            [1] * 4,  # res3 temporal kernel
            [3] * 6,  # res4 temporal kernel
            [3] * 3,  # res5 temporal kernel
        ],
        # temporal stride sizes
        [
            [1],  # conv1 temporal stride
            [1] * 3,  # res2 temporal stride
            [1] * 4,  # res3 temporal stride
            [1] * 6,  # res4 temporal stride
            [1] * 3,  # res5 temporal stride
        ],
        # temporal pool kernel sizes
        [2, 1],  # pool1 temporal kernel 1  # pool2 temporal kernel 2
    ),
}


class ResNet(nn.Module):
    """ResNet model."""

    def __init__(self, split):
        super(ResNet, self).__init__()
        self.split = split
        self._construct_imagenet()
        init_helper.init_weights(self)

    def _construct_imagenet(self):
        logger.info(
            "Constructing: ResNe(X)t-{}-{}x{}, {}, imagenet".format(
                cfg.RESNET.DEPTH,
                cfg.RESNET.NUM_GROUPS,
                cfg.RESNET.WIDTH_PER_GROUP,
                cfg.RESNET.TRANS_FUN,
            )
        )

        # Retrieve the number of blocks per stage (excluding base)
        (d2, d3, d4, d5) = _MODEL_STAGE_DS[cfg.RESNET.DEPTH]
        temp_kernel, temp_stride, temp_pool = _MODEL_STAGE_TEMPORAL[
            cfg.RESNET.ARCH
        ]

        # Compute the initial inner block dim
        num_groups = cfg.RESNET.NUM_GROUPS
        width_per_group = cfg.RESNET.WIDTH_PER_GROUP
        dim_inner = num_groups * width_per_group

        # Stage 1: (N, 3, 224, 224) -> (N, 64, 56, 56)
        self.s1 = stem_helper.ResStem(
            dim_in=3,
            dim_out=64,
            temp_kernel_size=temp_kernel[0][0],
            temp_stride_size=temp_stride[0][0],
            pool_size=temp_pool[0],
        )

        # Stage 2: (N, 64, 56, 56) -> (N, 256, 56, 56)
        self.s2 = resnet_helper.Res3DStage(
            dim_in=64,
            dim_out=256,
            stride=1,
            num_bs=d2,
            dim_inner=dim_inner,
            num_gs=num_groups,
            temp_kernel_sizes=temp_kernel[1],
            temp_stride_sizes=temp_stride[1],
        )

        # Temporal pooling to reduce computation
        if temp_pool[1] > 1:
            self.pool = nn.MaxPool3d(
                kernel_size=[temp_pool[1], 1, 1],
                stride=[temp_pool[1], 1, 1],
                padding=[0, 0, 0],
            )

        # Stage 3: (N, 256, 56, 56) -> (N, 512, 28, 28)
        self.s3 = resnet_helper.Res3DStage(
            dim_in=256,
            dim_out=512,
            stride=2,
            num_bs=d3,
            dim_inner=dim_inner * 2,
            num_gs=num_groups,
            temp_kernel_sizes=temp_kernel[2],
            temp_stride_sizes=temp_stride[2],
        )

        # Stage 4: (N, 512, 56, 56) -> (N, 1024, 14, 14)
        self.s4 = resnet_helper.Res3DStage(
            dim_in=512,
            dim_out=1024,
            stride=2,
            num_bs=d4,
            dim_inner=dim_inner * 4,
            num_gs=num_groups,
            temp_kernel_sizes=temp_kernel[3],
            temp_stride_sizes=temp_stride[3],
        )

        # Stage 5: (N, 1024, 14, 14) -> (N, 2048, 7, 7)
        self.s5 = resnet_helper.Res3DStage(
            dim_in=1024,
            dim_out=2048,
            stride=2,
            num_bs=d5,
            dim_inner=dim_inner * 8,
            num_gs=num_groups,
            temp_kernel_sizes=temp_kernel[4],
            temp_stride_sizes=temp_stride[4],
        )

        # Head: (N, 2048, 7, 7) -> (N, num_classes)
        self.head = head_helper.ResHead(
            dim_in=2048,
            num_classes=cfg.MODEL.NUM_CLASSES,
            temp_pools=temp_pool,
            temp_strides=temp_stride,
            split=self.split,
        )

    def forward(self, x):
        for module in self.children():
            x = module(x)
        return x
