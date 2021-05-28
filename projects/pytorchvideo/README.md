# Support PyTorchVideo in PySlowFast

## Introduction

[PyTorchVideo](https://pytorchvideo.org/) is a new deeplearning library with a focus on video understanding work, which provides reusable, modular and efficient components for video understanding. In PySlowFast, we add the support to incorporate PyTorchVideo components, including standard video datasets and state-of-the-art video models. Thus, we could use standard PySlowFast workflow to train and test PyTorchVideo datasets and models.

We add PySlowFast wrapper for different PyTorchVideo models and datasets. So we can easily construct PyTorchVideo datasets and models using PySlowFast config system. Right now, the supported [PyTorchVideo models](https://github.com/facebookresearch/SlowFast/blob/master/slowfast/models/ptv_model_builder.py) includes:
  * [I3D](https://arxiv.org/pdf/1705.07750.pdf)
  * [C2D](https://arxiv.org/pdf/1711.07971.pdf)
  * [R(2+1)D](https://openaccess.thecvf.com/content_cvpr_2018/papers/Tran_A_Closer_Look_CVPR_2018_paper.pdf)
  * [CSN](https://arxiv.org/abs/1904.02811)
  * [Slow, SlowFast](https://arxiv.org/pdf/1812.03982.pdf)
  * [X3D](https://arxiv.org/pdf/2004.04730.pdf)

The supported [PyTorchVideo datasets](https://github.com/facebookresearch/SlowFast/blob/master/slowfast/datasets/ptv_datasets.py) includes:
  * Kinetics
  * Charades
  * Something-something v2

## PyTorchVideo Model Zoo

We also provide a comprehensive PyTorchVideo Model Zoo using standard PySlowFast workflow and training recipe for PyTorchVideo datasets and models.


### Kinetics-400

| arch     | depth | pretrain | frame length x sample rate | top 1 | top 5 | Flops (G) x views | Params (M) | Model                                                                                                       | config                                         |
| -------- | ----- | -------- | -------------------------- | ----- | ----- | ----------------- | ---------- | ------------------------------------------------------------------------------------------------------------| ---------------------------------------------- |
| C2D      | R50   | \-       | 8x8                        | 71.46 | 89.68 | 25.89 x 3 x 10    | 24.33      | [link](https://dl.fbaipublicfiles.com/pytorchvideo/pysf_model_zoo/kinetics/C2D_8x8_R50.pyth)                | Kinetics/pytorchvideo/C2D_8x8_R50              |
| I3D      | R50   | \-       | 8x8                        | 73.27 | 90.70 | 37.53 x 3 x 10    | 28.04      | [link](https://dl.fbaipublicfiles.com/pytorchvideo/pysf_model_zoo/kinetics/I3D_8x8_R50.pyth)                | Kinetics/pytorchvideo/I3D_8x8_R50              |
| Slow     | R50   | \-       | 4x16                       | 72.40 | 90.18 | 27.55 x 3 x 10    | 32.45      | [link](https://dl.fbaipublicfiles.com/pytorchvideo/pysf_model_zoo/kinetics/SLOW_4x16_R50.pyth)              | Kinetics/pytorchvideo/SLOW_4x16_R50            |
| Slow     | R50   | \-       | 8x8                        | 74.58 | 91.63 | 54.52 x 3 x 10    | 32.45      | [link](https://dl.fbaipublicfiles.com/pytorchvideo/pysf_model_zoo/kinetics/SLOW_8x8_R50.pyth)               | Kinetics/pytorchvideo/SLOW_8x8_R50             |
| SlowFast | R50   | \-       | 4x16                       | 75.34 | 91.89 | 36.69 x 3 x 10    | 34.48      | [link](https://dl.fbaipublicfiles.com/pytorchvideo/pysf_model_zoo/kinetics/SLOWFAST_4x16_R50.pyth)          | Kinetics/pytorchvideo/SLOWFAST_4x16_R50        |
| SlowFast | R50   | \-       | 8x8                        | 76.94 | 92.69 | 65.71 x 3 x 10    | 34.57      | [link](https://dl.fbaipublicfiles.com/pytorchvideo/pysf_model_zoo/kinetics/SLOWFAST_8x8_R50.pyth)           | Kinetics/pytorchvideo/SLOWFAST_8x8_R50         |
| SlowFast | R101  | \-       | 8x8                        | 77.90 | 93.27 | 127.20 x 3 x 10   | 62.83      | [link](https://dl.fbaipublicfiles.com/pytorchvideo/pysf_model_zoo/kinetics/SLOWFAST_8x8_R101.pyth)          | Kinetics/pytorchvideo/SLOWFAST_8x8_R101        |
| SlowFast | R101  | \-       | 16x8                       | 78.70 | 93.61 | 215.61 x 3 x 10   | 53.77      | [link](https://dl.fbaipublicfiles.com/pytorchvideo/pysf_model_zoo/kinetics/SLOWFAST\_16x8\_R101_50_50.pyth) | Kinetics/pytorchvideo/SLOWFAST_16x8_R101_50_50 |
| CSN      | R101  | \-       | 32x2                       | 77.00 | 92.90 | 75.62 x 3 x 10    | 22.21      | [link](https://dl.fbaipublicfiles.com/pytorchvideo/pysf_model_zoo/kinetics/CSN_32x2_R101.pyth)              | Kinetics/pytorchvideo/CSN_32x2_R101            |
| R(2+1)D  | R50   | \-       | 16x4                       | 76.01 | 92.23 | 76.45 x 3 x 10    | 28.11      | [link](https://dl.fbaipublicfiles.com/pytorchvideo/pysf_model_zoo/kinetics/R2PLUS1D_16x4_R50.pyth)          | Kinetics/pytorchvideo/R2PLUS1D_16x4_R50        |
| X3D      | XS    | \-       | 4x12                       | 69.12 | 88.63 | 0.91 x 3 x 10     | 3.79       | [link](https://dl.fbaipublicfiles.com/pytorchvideo/pysf_model_zoo/kinetics/X3D_XS.pyth)                     | Kinetics/pytorchvideo/X3D_XS                   |
| X3D      | S     | \-       | 13x6                       | 73.33 | 91.27 | 2.96 x 3 x 10     | 3.79       | [link](https://dl.fbaipublicfiles.com/pytorchvideo/pysf_model_zoo/kinetics/X3D_S.pyth)                      | Kinetics/pytorchvideo/X3D_S                    |
| X3D      | M     | \-       | 16x5                       | 75.94 | 92.72 | 6.72 x 3 x 10     | 3.79       | [link](https://dl.fbaipublicfiles.com/pytorchvideo/pysf_model_zoo/kinetics/X3D_M.pyth)                      | Kinetics/pytorchvideo/X3D_M                    |
| X3D      | L     | \-       | 16x5                       | 77.44 | 93.31 | 26.64 x 3 x 10    | 6.15       | [link](https://dl.fbaipublicfiles.com/pytorchvideo/pysf_model_zoo/kinetics/X3D_L.pyth)                      | Kinetics/pytorchvideo/X3D_L                    |


### Something-Something V2

| arch     | depth | pretrain     | frame length x sample rate | top 1 | top 5 | Flops (G) x views | Params (M) | Model                                                                                               | config             |
| -------- | ----- | ------------ | -------------------------- | ----- | ----- | ----------------- | ---------- | --------------------------------------------------------------------------------------------------- | ------------------ |
| Slow     | R50   | Kinetics 400 | 8x8                        | 60.04 | 85.19 | 55.10 x 3 x 1     | 31.96      | [link](https://dl.fbaipublicfiles.com/pytorchvideo/pysf_model_zoo/ssv2/SLOW_8x8_R50.pyth)     | SSv2/pytorchvideo/SLOW_8x8_R50     |
| SlowFast | R50   | Kinetics 400 | 8x8                        | 61.68 | 86.92 | 66.60 x 3 x 1     | 34.04      | [link](https://dl.fbaipublicfiles.com/pytorchvideo/pysf_model_zoo/ssv2/SLOWFAST_8x8_R50.pyth) | SSv2/pytorchvideo/SLOWFAST_8x8_R50 |

### Charades

| arch     | depth | pretrain     | frame x interval | MAP   | Flops (G) x views | Params (M) | Model                                                                                               | config             |
| -------- | ----- | ------------ | ---------------- | ----- | ----------------- | ---------- | --------------------------------------------------------------------------------------------------- | ------------------ |
| Slow     | R50   | Kinetics 400 | 8x8              | 34.72 | 55.10 x 3 x 10    | 31.96      | [link](https://dl.fbaipublicfiles.com/pytorchvideo/pysf_model_zoo/ssv2/SLOW_8x8_R50.pyth)     | Charades/pytorchvideo/SLOW_8x8_R50     |
| SlowFast | R50   | Kinetics 400 | 8x8              | 37.24 | 66.60 x 3 x 10    | 34.00      | [link](https://dl.fbaipublicfiles.com/pytorchvideo/pysf_model_zoo/ssv2/SLOWFAST_8x8_R50.pyth) | Charades/pytorchvideo/SLOWFAST_8x8_R50 |

Notes:
* The above model weights has slightly difference with these in [PyTorchVideo official model zoo](https://github.com/facebookresearch/pytorchvideo/blob/master/docs/source/model_zoo.md). The layer names of above model weights will contain the additional prefix of `model.` due to the [model wrapper](https://github.com/facebookresearch/SlowFast/blob/master/slowfast/models/ptv_model_builder.py) in PySlowFast.
* For `Flops x views` column, we report the inference cost with a single “view" × the number of views (FLOPs × space_views × time_views). For example, we take 3 spatial crops for 10 temporal clips on Kinetics.
