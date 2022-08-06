# Masked Feature Prediction for Self-Supervised Visual Pre-Training
[Chen Wei*](https://weichen582.github.io/), [Haoqi Fan](https://haoqifan.github.io/), [Saining Xie](https://www.sainingxie.com/), [Chao-Yuan Wu](https://chaoyuan.org/), [Alan Yuille](https://www.cs.jhu.edu/~ayuille/), [Christoph Feichtenhofer*](http://feichtenhofer.github.io/)
<br/>
In CVPR, 2022. [[Paper](https://arxiv.org/abs/2112.09133)]
<br/>
<div align="center">
  <img src="http://feichtenhofer.github.io/pubs/maskfeat_concept.png" width="500px">
</div>
<br/>

## Results & Models

### **ImageNet-1K**; configs are under configs/masked_ssl/


| name | top1 |  config pre-train (PT) | config fine-tune | model PT |
| ------------- | ------------- | ------------- | ------------- | ------------- |
| ViT-B | 84.0 |  in1k_VIT_B_MaskFeat_PT | in1k_VIT_B_MaskFeat_FT | [`link`](https://dl.fbaipublicfiles.com/pyslowfast/masked_models/in1k_VIT_B_MaskFeat_PT_epoch_01600.pyth) |
| ViT-L | 85.7 |  in1k_VIT_L_MaskFeat_PT | in1k_VIT_L_MaskFeat_FT | [`link`](https://dl.fbaipublicfiles.com/pyslowfast/masked_models/in1k_VIT_L_MaskFeat_PT_epoch_01600.pyth) |



### **Kinetics-400**; configs are under configs/masked_ssl/


| name | frame length x sample rate | top1 |  Flops (G) x views | #params (M) |   config pre-train (PT) | config fine-tune | model PT |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| MViT-S | 16 x 4 | 82.2 | 71 x 1 x 10 | 36 |  k400_MVITv2_S_16x4_MaskFeat_PT | k400_MVITv2_S_16x4_FT | [`link`](https://dl.fbaipublicfiles.com/pyslowfast/masked_models/k400_MVIT_S_MaskFeat_PT_epoch_00300.pyth) |
| MViT-L | 16 x 4 | 84.3 | 377 x 1 x 10 | 218 |  k400_MVITv2_L_16x4_MaskFeat_PT | k400_MVITv2_L_16x4_FT | [`link`](https://dl.fbaipublicfiles.com/pyslowfast/masked_models/k400_MVIT_L_MaskFeat_PT_epoch_00800.pyth) |



## Getting started
To use self-supervised learning techniques please refer to the configs under `configs/masked_ssl`. For example, the command

```
python tools/run_net.py \
  --cfg configs/masked_ssl/k400_MVITv2_L_16x4_MaskFeat_PT.yaml \
  DATA.PATH_TO_DATA_DIR path_to_your_Kinetics_dataset
```

should train a MaskFeat MViT-L model on the Kinetics-400 dataset, and the command

```
python tools/run_net.py \
  --cfg configs/masked_ssl/k400_MVITv2_L_16x4_FT.yaml \
  DATA.PATH_TO_DATA_DIR path_to_your_Kinetics_dataset \
  TRAIN.CHECKPOINT_FILE_PATH path_to_your_pretrain_checkpoint
```

will fine-tune the resulting model, after passing the checkpoint path to the config.

For images, the command

```
python tools/run_net.py \
  --cfg configs/masked_ssl/in1k_VIT_B_MaskFeat_PT.yaml \
  DATA.PATH_TO_DATA_DIR path_to_your_ImageNet_dataset
```

should train a MaskFeat ViT-B model on the ImageNet dataset, and the command

```
python tools/run_net.py \
  --cfg configs/masked_ssl/in1k_VIT_B_FT.yaml \
  DATA.PATH_TO_DATA_DIR path_to_your_ImageNet_dataset \
  TRAIN.CHECKPOINT_FILE_PATH path_to_your_pretrain_checkpoint
```

will fine-tune the resulting model, after passing the checkpoint path to the config.

## Reference
If you find this useful for your research, please consider citing the paper using the following BibTeX entry.
```BibTeX
@InProceedings{wei2022masked,
    author    = {Wei, Chen and Fan, Haoqi and Xie, Saining and Wu, Chao-Yuan and Yuille, Alan and Feichtenhofer, Christoph},
    title     = {Masked Feature Prediction for Self-Supervised Visual Pre-Training},
    booktitle = {CVPR},
    year      = {2022},
}
```
