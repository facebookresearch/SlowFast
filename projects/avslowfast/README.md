# Getting Started with PyAVSlowFast

This section supplements the original doc in PySlowFast (attached below) and provide instructions on how to start training AVSlowFast model with this codebase. 

First, a note that `DATA.PATH_TO_DATA_DIR` points to the directory where annotation csv files reside and `DATA.PATH_PREFIX` to the root of the data directory. 

Then, issue the following training command
```
python tools/run_net.py \
  --cfg configs/Kinetics/AVSLOWFAST_4x16_R50.yaml \
  DATA.PATH_TO_DATA_DIR path_to_your_annotation \  
  DATA.PATH_PREFIX path_to_your_dataset_root \ 
  NUM_GPUS 8 \
  DATA_LOADER.NUM_WORKERS 8 \
  TRAIN.BATCH_SIZE 64 \
```

For testing, run the following
```
python tools/run_net.py \
  --cfg configs/Kinetics/AVSLOWFAST_4x16_R50.yaml \
  DATA.PATH_TO_DATA_DIR path_to_your_annotation \  
  DATA.PATH_PREFIX path_to_your_dataset_root \ 
  TEST.BATCH_SIZE 32 \
  TEST.CHECKPOINT_FILE_PATH path_to_your_checkpoint \
  TRAIN.ENABLE False \
```
