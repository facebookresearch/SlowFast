# /bin/bash

WORK_SPACE=$HOME/slowfast/
cd $WORK_SPACE
# cd "$HOME/dataset/"
echo $PWD

CUDA_VISIBLE_DEVICES=1 /home/hong/miniconda3/envs/slowfast/bin/python tools/run_net.py --cfg configs/contrastive_ssl/custom_BYOL_SlowR50_8x8.yaml
# CUDA_VISIBLE_DEVICES=1 python tools/run_net.py --cfg configs/contrastive_ssl/custom_BYOL_SlowR50_8x8.yaml
