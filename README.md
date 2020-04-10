# PySlowFast

PySlowFast is an open source video understanding codebase from FAIR that provides state-of-the-art video classification models, including papers "[SlowFast Networks for Video Recognition](https://arxiv.org/abs/1812.03982)", and "[Non-local Neural Networks](https://arxiv.org/abs/1711.07971)". 

<div align="center">
  <img src="demo/ava_demo.gif" width="600px"/>
</div>

## Introduction

The goal of PySlowFast is to provide a high-performance, light-weight pytorch codebase provides state-of-the-art video backbones for video understanding research on different tasks (classification, detection, and etc). It is designed in order to support rapid implementation and evaluation of novel video research ideas. PySlowFast includes implementations of the following backbone network architectures:

- SlowFast 
- Slow
- C2D
- I3D
- Non-local Network

## Updates

PySlowFast is released in conjunction with our [ICCV 2019 Tutorial](https://alexander-kirillov.github.io/tutorials/visual-recognition-iccv19/).

## License

PySlowFast is released under the [Apache 2.0 license](LICENSE). 

## Model Zoo and Baselines

We provide a large set of baseline results and trained models available for download in the PySlowFast [Model Zoo](MODEL_ZOO.md).

## Installation

Please find installation instructions for PyTorch and PySlowFast in [INSTALL.md](INSTALL.md). You may follow the instructions in [DATASET.md](slowfast/datasets/DATASET.md) to prepare the datasets.

## Quick Start

Follow the example in [GETTING_STARTED.md](GETTING_STARTED.md) to start playing video models with PySlowFast.

## Contributors
PySlowFast is written and maintained by [Haoqi Fan](https://haoqifan.github.io/), [Yanghao Li](https://lyttonhao.github.io/), [Wan-Yen Lo](https://www.linkedin.com/in/wanyenlo/), [Christoph Feichtenhofer](https://feichtenhofer.github.io/).
