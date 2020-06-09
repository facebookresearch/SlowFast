# A Multigrid Method for Efficiently Training Video Models
[Chao-Yuan Wu](https://www.cs.utexas.edu/~cywu/),
[Ross Girshick](http://rossgirshick.info),
[Kaiming He](http://kaiminghe.com),
[Christoph Feichtenhofer](http://feichtenhofer.github.io/),
[Philipp Kr&auml;henb&uuml;hl](http://www.philkr.net/)
<br/>
In CVPR, 2020. [[Paper](https://arxiv.org/abs/1912.00998)]
<br/>
<div align="center">
  <img src="multigrid.png" width="700px" />
</div>
<br/>


## Getting started
To enable multigrid training, add `MULTIGRID.LONG_CYCLE True` and/or `MULTIGRID.SHORT_CYCLE True` when training your model. (Default multigrid training uses both long and short cycles; See [paper](https://arxiv.org/abs/1912.00998) for details.) For example,

```
python tools/run_net.py \
  --cfg configs/Charades/SLOWFAST_16x8_R50.yaml \
  DATA.PATH_TO_DATA_DIR path_to_your_dataset \
  MULTIGRID.LONG_CYCLE True \
  MULTIGRID.SHORT_CYCLE True \
```
This should train multiple times faster than training *without* multigrid training.
Note that multigrid training might induce higher IO overhead.
Systems with faster IO (e.g., with efficient local disk) might enjoy more speedup.
Please see [MODEL_ZOO.md](https://github.com/facebookresearch/SlowFast/blob/master/MODEL_ZOO.md) for more examples of multigrid training.

## Citing Multigrid Training
If you use multigrid training or the models from MODEL_ZOO in your research, please use the following BibTeX entry.
```BibTeX
@inproceedings{multigrid2020,
  Author    = {Chao-Yuan Wu and Ross Girshick and Kaiming He and Christoph Feichtenhofer
               and Philipp Kr\"{a}henb\"{u}hl},
  Title     = {{A Multigrid Method for Efficiently Training Video Models}},
  Booktitle = {{CVPR}},
  Year      = {2020}}
```
