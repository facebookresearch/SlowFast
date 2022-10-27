import os

os.system(
    "taskset -c 0-11 /home/hong/miniconda3/envs/slowfast/bin/python tools/run_net.py --cfg configs/contrastive_ssl/custom_BYOL_SlowR50_8x8.yaml"
)
