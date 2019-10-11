# SlowFast Video Understanding

SlowFast Video Understanding is an open source codebase from FAIR that reproduces state-of-the-art video classification models, including papers "[SlowFast Networks for Video Recognition](https://arxiv.org/abs/1812.03982)", and "[Non-local Neural Networks](https://arxiv.org/abs/1711.07971)". The codebase also reproduces state-of-the-art video detection results that won the [1st place] (https://static.googleusercontent.com/media/research.google.com/en//ava/2019/fair_slowfast.pdf) on the AVA action detection track. 

## Introduction

The goal of SlowFast Video Understanding is to provide a high-performance, light-weight pytorch codebase reproducing state-of-the-art video backbones for video understanding research. It is designed in order to support rapid implementation and evaluation of novel video research ideas. SlowFast Video Understanding includes implementations of the following backbone network architectures:

- SlowFast and SlowOnly
- C2D, I3D, and Non-local Network

## Update

SlowFast Video Understanding is released in conjunction with our [ICCV 2019 Tutorial](https://alexander-kirillov.github.io/tutorials/visual-recognition-iccv19/). The slides can be found [here](TODO).

## License

SlowFast Video Understanding is released under the [Apache 2.0 license](https://github.com/facebookresearch/detectron/blob/master/LICENSE). See the [NOTICE](NOTICE) file for additional details.

## Model Zoo and Baselines

We provide a large set of baseline results and trained models available for download in the SlowFast Video Understanding [Model Zoo](MODEL_ZOO.md).

## Installation

Please find installation instructions for Pytorch and SlowFast Video Understanding in [INSTALL.md](MODEL_ZOO.md).

## Quick Start: Using SlowFast Video Understanding

Install slowfast Video Understanding and train your first state-of-the-art model. Please see [GETTING_STARTED.md](GETTING_STARTED.md) for brief tutorial.

## Quick Start: Using SlowFast Video Understanding as A Library

Install slowfast Video Understanding as a feature extractor for state-of-the-art backbones. Please see [SLOWFAST_AS_LIBRARY.md](SLOWFAST_AS_LIBRARY.md) for brief tutorial.
