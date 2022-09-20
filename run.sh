# /bin/bash

WORK_SPACE=$HOME/slowfast/
cd $WORK_SPACE
# cd "$HOME/dataset/"
echo $PWD


python tools/run_net.py --cfg configs/contrastive_ssl/custom_BYOL_SlowR50_8x8.yaml

# check another metrics for this such as epochs, num_workers, batch_size, and learning_rate.
