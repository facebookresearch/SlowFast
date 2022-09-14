# /bin/bash

WORK_SPACE=$HOME/slowfast/
cd $WORK_SPACE
# cd "$HOME/dataset/"
echo $PWD

# variables for config
DATASET_PATH=/data/hong/k400/reduced
BATCH_SIZE=16
NUM_WORKERS=10

python tools/run_net.py \
    --cfg configs/Kinetics/contrastive_ssl/MoCo_SlowR50_8x8.yaml \
	DATA.PATH_TO_DATA_DIR $DATASET_PATH \
    DATA_LOADER.NUM_WORKERS $NUM_WORKERS \
    NUM_GPUS 1 \
    TRAIN.BATCH_SIZE $BATCH_SIZE \

# check another metrics for this such as epochs, num_workers, batch_size, and learning_rate.
