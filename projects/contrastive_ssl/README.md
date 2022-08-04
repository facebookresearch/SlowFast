# A Large-Scale Study on Unsupervised Spatiotemporal Representation Learning
[Christoph Feichtenhofer](http://feichtenhofer.github.io/), [Haoqi Fan](https://haoqifan.github.io/), [Bo Xiong](https://www.cs.utexas.edu/~bxiong/), [Ross Girshick](http://www.cs.berkeley.edu/~rbg/), [Kaiming He](http://kaiminghe.com/)
<br/>
In CVPR, 2021. [[Paper](https://arxiv.org/abs/2104.14558)]
<br/>
<div align="center">
  <img src="http://feichtenhofer.github.io/pubs/videomoco_concept2.png" width="500px">
</div>
<br/>


## Kinetics 400 and 600

| method | &rho;| architecture | size |  frames x sampling |  pretrain data | K400-linear |  UCF101-split1  | AVA  | SSv2  | model | config |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| MoCo | 2 | Slow-only | R50 | 8 x 8 | Kinetics-400 | 66.6 | 91.3  | 19.7 | 52.7 | [`link`](https://dl.fbaipublicfiles.com/pyslowfast/videomoco_models/MoCo_SlowR50_8x8_T2_epoch_00200.pyth) | contrastive_ssl/MoCo_SlowR50_8x8 |
| BYOL | 2 | Slow-only | R50 | 8 x 8 | Kinetics-400 | 67.4 | 94.0  | 22.8 | 54.4 | [`link`](https://dl.fbaipublicfiles.com/pyslowfast/videomoco_models/BYOL_SlowR50_8x8_T2_epoch_00200.pyth) | contrastive_ssl/BYOL_SlowR50_8x8 |
| SimCLR | 2 | Slow-only | R50 | 8 x 8 | Kinetics-400 | 61.5 | 88.3  | 17.5 | 51.4 | [`link`](https://dl.fbaipublicfiles.com/pyslowfast/videomoco_models/SimCLR_SlowR50_8x8_T2_epoch_00200.pyth) | contrastive_ssl/SimCLR_SlowR50_8x8 |
| SwAV | 2 | Slow-only | R50 | 8 x 8 | Kinetics-400 |  62.6 | 90.2  | 19.2 | 52.5| [`link`](https://dl.fbaipublicfiles.com/pyslowfast/videomoco_models/SwAV_SlowR50_8x8_T2_epoch_00200.pyth) | contrastive_ssl/SwAV_SlowR50_8x8 |
| MoCo | 4 | Slow-only | R50 | 8 x 8 | Kinetics-400 | 71.0 | 94.5  | 21.9 | 54.0 |  [`link`](https://dl.fbaipublicfiles.com/pyslowfast/videomoco_models/MoCo_SlowR50_8x8_T4_epoch_00200.pyth) | contrastive_ssl/MoCo_SlowR50_8x8 |
| BYOL | 4 | Slow-only | R50 | 8 x 8 | Kinetics-400 | 70.1 | 94.7 | xx | xx | [`link`](https://dl.fbaipublicfiles.com/pyslowfast/videomoco_models/BYOL_SlowR50_8x8_T4_epoch_00200.pyth) | contrastive_ssl/BYOL_SlowR50_8x8 |
| BYOL | 4 | Slow-only | R50 | 16 x 4 | Kinetics-400 | 71.1 | 95.4  | xx | xx | [`link`](https://dl.fbaipublicfiles.com/pyslowfast/videomoco_models/BYOL_SlowR50_16x4_T4_epoch_00200.pyth) | contrastive_ssl/BYOL_SlowR50_8x8 |


## Getting started
To use self-supervised learning techniques please refer to the configs under `configs/contrastive_ssl`, or see the [MODEL_ZOO.md](https://github.com/facebookresearch/SlowFast/blob/master/MODEL_ZOO.md) for pre-trained models. See [paper](https://arxiv.org/abs/2104.14558) for details. For example, the command

```
python tools/run_net.py \
  --cfg configs/Kinetics/contrastive_ssl/MoCo_SlowR50_8x8.yaml \
  DATA.PATH_TO_DATA_DIR path_to_your_dataset \
```

should train a MoCo R50 Slow-only model with 8x8 sampling on your dataset.


## Reference
If you find this useful for your research, please consider citing the paper using the following BibTeX entry.
```BibTeX
@inproceedings{videossl2021,
  Author    = {Christoph Feichtenhofer, Haoqi Fan, Bo Xiong, Ross Girshick, Kaiming He},
  Title     = {A Large-Scale Study on Unsupervised Spatiotemporal Representation Learning},
  Booktitle = {CVPR},
  Year      = {2021}}
```
