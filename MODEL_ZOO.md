# PySlowFast Model Zoo and Baselines

## Kinetics

We provided original pretrained models from Caffe2 on heavy models (testing Caffe2 pretrained model in PyTorch might have a small different in performance):

| architecture | depth |  pretrain |  frame length x sample rate | top1 |  top5  |  model | config |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| C2D | R50 | - | 8 x 8 | 67.2 | 87.8 | [`link`](https://dl.fbaipublicfiles.com/pyslowfast/model_zoo/kinetics400/C2D_NOPOOL_8x8_R50.pkl) | Kinetics/c2/C2D_NOPOOL_8x8_R50 |
| I3D | R50 | - | 8 x 8 | 73.5 | 90.8 | [`link`](https://dl.fbaipublicfiles.com/pyslowfast/model_zoo/kinetics400/I3D_8x8_R50.pkl) | Kinetics/c2/I3D_8x8_R50 |
| I3D NLN | R50 | - | 8 x 8 | 74.0 | 91.1 | [`link`](https://dl.fbaipublicfiles.com/pyslowfast/model_zoo/kinetics400/I3D_NLN_8x8_R50.pkl) | Kinetics/c2/I3D_NLN_8x8_R50 |
| Slow | R50 | - | 4 x 16 | 72.7 | 90.3 | [`link`](https://dl.fbaipublicfiles.com/pyslowfast/model_zoo/kinetics400/SLOWONLY_4x16_R50.pkl) | Kinetics/c2/SLOW_4x16_R50 |
| Slow | R50 | - | 8 x 8 | 74.8 | 91.6 | [`link`](https://dl.fbaipublicfiles.com/pyslowfast/model_zoo/kinetics400/SLOWONLY_8x8_R50.pkl) | Kinetics/c2/SLOW_8x8_R50 |
| SlowFast | R50 | - | 4 x 16 | 75.6 | 92.0 | [`link`](https://dl.fbaipublicfiles.com/pyslowfast/model_zoo/kinetics400/SLOWFAST_4x16_R50.pkl) | Kinetics/c2/SLOWFAST_4x16_R50 |
| SlowFast | R50 | - | 8 x 8 | 77.0 | 92.6 | [`link`](https://dl.fbaipublicfiles.com/pyslowfast/model_zoo/kinetics400/SLOWFAST_8x8_R50.pkl) | Kinetics/c2/SLOWFAST_8x8_R50 |
| SlowFast | R101 | - | 8 x 8 | 78.0 | 93.3 | [`link`](coming_soon) | Kinetics/c2/SLOWFAST_8x8_R101_101_101|
| SlowFast | R101 | - | 16 x 8 | 78.9 | 93.5 | [`link`](coming_soon) | Kinetics/c2/SLOWFAST_16x8_R101_50_50 |

## X3D models (details in projects/x3d)

|    architecture     |  size  | pretrain |    frame length x sample rate     | top1 10-view | top1 30-view | parameters (M) | FLOPs (G) | model | config |
| :-------------: | :-----: | :-----: | :-------------: | :------: | :------: | :------------: | :----: | :------: | :------: |
| X3D | XS | - | 4 x 12 | 68.7 | 69.5 | 3.8 | 0.60 | [`link`](https://dl.fbaipublicfiles.com/pyslowfast/x3d_models/x3d_xs.pyth) | Kinetics/X3D_XS |
| X3D | S | - | 13 x 6 | 73.1 | 73.5 | 3.8 | 1.96 | [`link`](https://dl.fbaipublicfiles.com/pyslowfast/x3d_models/x3d_s.pyth) | Kinetics/X3D_S |
| X3D | M | - | 16 x 5 | 75.1 | 76.2 | 3.8 | 4.73 | [`link`](https://dl.fbaipublicfiles.com/pyslowfast/x3d_models/x3d_m.pyth) | Kinetics/X3D_M |
| X3D | L | - | 16 x 5 | 76.9 | 77.5 | 6.2 | 18.37 | [`link`](https://dl.fbaipublicfiles.com/pyslowfast/x3d_models/x3d_l.pyth) | Kinetics/X3D_L |

## AVA

| architecture | depth | Pretrain Model |  frame length x sample rate  | MAP | AVA version | model |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |------------- |
| Slow | R50 | [Kinetics 400](https://dl.fbaipublicfiles.com/pyslowfast/model_zoo/ava/pretrain/C2D_8x8_R50.pkl) | 4 x 16 | 19.5 | 2.2 | [`link`](https://dl.fbaipublicfiles.com/pyslowfast/model_zoo/ava/C2D_8x8_R50.pkl) |
| SlowFast | R101 | [Kinetics 600](https://dl.fbaipublicfiles.com/pyslowfast/model_zoo/ava/pretrain/SLOWFAST_32x2_R101_50_50_v2.1.pkl) | 8 x 8 | 28.2 | 2.1 | [`link`](https://dl.fbaipublicfiles.com/pyslowfast/model_zoo/ava/SLOWFAST_32x2_R101_50_50_v2.1.pkl) |
| SlowFast | R101 | [Kinetics 600](https://dl.fbaipublicfiles.com/pyslowfast/model_zoo/ava/pretrain/SLOWFAST_32x2_R101_50_50.pkl) | 8 x 8 | 29.1 | 2.2 | [`link`](https://dl.fbaipublicfiles.com/pyslowfast/model_zoo/ava/SLOWFAST_32x2_R101_50_50.pkl) |
| SlowFast | R101 | [Kinetics 600](https://dl.fbaipublicfiles.com/pyslowfast/model_zoo/ava/pretrain/SLOWFAST_64x2_R101_50_50.pkl) | 16 x 8 | 29.4 | 2.2 | [`link`](https://dl.fbaipublicfiles.com/pyslowfast/model_zoo/ava/SLOWFAST_64x2_R101_50_50.pkl) |

## Multigrid Training

***Update June, 2020:*** In the following we provide (reimplemented) models from  "[A Multigrid Method for Efficiently Training Video Models
](https://arxiv.org/abs/1912.00998)" paper. The multigrid method trains about 3-6x faster than the original training on multiple datasets. See [projects/multigrid](projects/multigrid/README.md) for more information. The following provides models, results, and example config files.

#### Kinetics:
| architecture | depth |  pretrain |  frame length x sample rate | training | top1 |  top5  |  model | config |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| SlowFast | R50 | - | 8 x 8 | Standard | 76.8 | 92.7 | [`link`](https://dl.fbaipublicfiles.com/pyslowfast/pyslowfast/model_zoo/multigrid/model_zoo/Kinetics/SLOWFAST_8x8_R50_stepwise.pkl) | Kinetics/SLOWFAST_8x8_R50_stepwise |
| SlowFast | R50 | - | 8 x 8 | Multigrid | 76.6 | 92.7 | [`link`](https://dl.fbaipublicfiles.com/pyslowfast/pyslowfast/model_zoo/multigrid/model_zoo/Kinetics/SLOWFAST_8x8_R50_stepwise_multigrid.pkl) | Kinetics/SLOWFAST_8x8_R50_stepwise_multigrid |

(Here we use stepwise learning rate schedule.)

#### Something-Something V2:
| architecture | depth |  pretrain |  frame length x sample rate | training | top1 |  top5  |  model | config |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| SlowFast | R50 | [Kinetics 400](https://dl.fbaipublicfiles.com/pyslowfast/model_zoo/kinetics400/SLOWFAST_8x8_R50.pkl) | 16 x 8 | Standard | 63.0 | 88.5 | [`link`](https://dl.fbaipublicfiles.com/pyslowfast/pyslowfast/model_zoo/multigrid/model_zoo/SSv2/SLOWFAST_16x8_R50.pkl) | SSv2/SLOWFAST_16x8_R50 |
| SlowFast | R50 | [Kinetics 400](https://dl.fbaipublicfiles.com/pyslowfast/model_zoo/kinetics400/SLOWFAST_8x8_R50.pkl) | 16 x 8 | Multigrid | 63.5 | 88.7 | [`link`](https://dl.fbaipublicfiles.com/pyslowfast/pyslowfast/model_zoo/multigrid/model_zoo/SSv2/SLOWFAST_16x8_R50_multigrid.pkl) | SSv2/SLOWFAST_16x8_R50_multigrid |


#### Charades
| architecture | depth |  pretrain |  frame length x sample rate | training | mAP |  model | config |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| SlowFast | R50 | [Kinetics 400](https://dl.fbaipublicfiles.com/pyslowfast/model_zoo/kinetics400/SLOWFAST_8x8_R50.pkl) | 16 x 8 | Standard | 38.9 | [`link`](https://dl.fbaipublicfiles.com/pyslowfast/pyslowfast/model_zoo/multigrid/model_zoo/Charades/SLOWFAST_16x8_R50.pkl) | SSv2/SLOWFAST_16x8_R50 |
| SlowFast | R50 | [Kinetics 400](https://dl.fbaipublicfiles.com/pyslowfast/model_zoo/kinetics400/SLOWFAST_8x8_R50.pkl) | 16 x 8 | Multigrid | 38.6 | [`link`](https://dl.fbaipublicfiles.com/pyslowfast/pyslowfast/model_zoo/multigrid/model_zoo/Charades/SLOWFAST_16x8_R50_multigrid.pkl) | SSv2/SLOWFAST_16x8_R50_multigrid |


## ImageNet

We also release the imagenet pretrained model if finetuning from ImageNet is preferred. The reported accuracy is obtained by center crop testing on the validation set.

| architecture | depth |  Top1 |  Top5  |  model  |
| ------------- | ------------- | ------------- | ------------- | ------------- |
| ResNet | R50 | 23.6 | 6.8 | [`link`](https://dl.fbaipublicfiles.com/pyslowfast/model_zoo/kinetics400/R50_IN1K.pyth) |
