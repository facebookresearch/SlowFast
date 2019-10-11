#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from setuptools import find_packages, setup

setup(
    name="slowfast",
    version="1.0",
    author="FAIR",
    url="unknown",
    description="SlowFast Video Understanding",
    install_requires=[
        "yacs>=0.1.6",
        "pyyaml>=5.1",
        "fvcore",
        "pyav",
        "matplotlib",
        "yacs>=0.1.6",
        "termcolor>=1.1",
    ],
    packages=find_packages(exclude=("configs", "tests")),
)
