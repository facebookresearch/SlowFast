#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from .build import MODEL_REGISTRY, build_model  # noqa
from .contrastive import ContrastiveModel  # noqa
from .custom_video_model_builder import *  # noqa
from .video_model_builder import ResNet, SlowFast  # noqa

from .shuffleNetV2 import SlowFastShuffleNetV2
from .mobileNetV2 import MobileNetV2
from .squeezeNet import Squeezenet
from .shuffleNetV2R2Plus1 import SlowFastShuffleNetV2R2Plus1
from .shuffleNetV2D2 import SlowFastShuffleNetV2D2
from .videoTransformer import VisionTransformer

try:
    from .ptv_model_builder import (
        PTVCSN,
        PTVX3D,
        PTVR2plus1D,
        PTVResNet,
        PTVSlowFast,
    )  # noqa
except Exception:
    print("Please update your PyTorchVideo to latest master")
