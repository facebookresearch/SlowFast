#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from setuptools import find_packages, setup
import re

# Check dependencies
import torchvision
import cv2
import detectron2
torchvision_ver = re.match(r'(?:\d+!)?([\.\d]+)', torchvision.__version__)[1]
assert [int(x) for x in torchvision_ver.split(".")] >= [0, 4, 2]


setup(
    name="slowfast",
    version="1.0",
    author="FAIR",
    url="https://github.com/facebookresearch/SlowFast",
    description="SlowFast Video Understanding",
    install_requires=[
        # These dependencies are not pure-python.
        # In general, avoid adding dependencies that are not pure-python because they are not
        # guaranteed to be installable by `pip install` on all platforms.
        "Pillow",  # or use pillow-simd for better performance
        # Do not add opencv here. Just like pytorch, user should install
        # opencv themselves, preferrably by OS's package manager, or by
        # choosing the proper pypi package name at https://github.com/skvark/opencv-python
        # Also, avoid adding dependencies that transitively depend on pytorch or opencv.
        # ------------------------------------------------------------
        # The following are pure-python dependencies that should be easily installable.
        # But still be careful when adding more: fewer people are able to use the software
        # with every new dependency added.
        "yacs>=0.1.6",
        "pyyaml>=5.1",
        "av",
        "matplotlib",
        "termcolor>=1.1",
        "simplejson",
        "tqdm",
        "psutil",
        "matplotlib",
        "pandas",
        "scikit-learn",
        "tensorboard",
        "fairscale",
    ],
    extras_require={"tensorboard_video_visualization": ["moviepy"]},
    packages=find_packages(exclude=("configs", "tests")),
)
