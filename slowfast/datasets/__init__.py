#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from .ava_dataset import Ava  # noqa
from .build import build_dataset, DATASET_REGISTRY  # noqa
from .charades import Charades  # noqa
from .imagenet import Imagenet  # noqa
from .kinetics import Kinetics  # noqa
from .ssv2 import Ssv2  # noqa
from .my_custom_dataset import Custom

try:
    from .ptv_datasets import Ptvcharades, Ptvkinetics, Ptvssv2  # noqa
except Exception:
    print("Please update your PyTorchVideo to latest master")
