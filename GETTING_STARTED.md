# Getting Started with PySlowFast

This document provides a brief intro of launching jobs in PySlowFast for training and testing. Before launching any job, make sure you have properly installed the PySlowFast following the instruction in [README.md](README.md) and you have prepared the dataset following [DATASET.md](slowfast/datasets/DATASET.md) with the correct format.

## Train a Standard Model from Scratch

Here we can start with training a simple C2D models by running:

```
python tools/run_net.py \
  --cfg configs/Kinetics/C2D_8x8_R50.yaml \
  DATA.PATH_TO_DATA_DIR path_to_your_dataset \
  NUM_GPUS 2 \
  TRAIN.BATCH_SIZE 16 \
```
You may need to pass location of your dataset in the command line by adding `DATA.PATH_TO_DATA_DIR path_to_your_dataset`, or you can simply add 

```
DATA:
  PATH_TO_DATA_DIR: path_to_your_dataset
```
To the yaml configs file, then you do not need to pass it to the command line every time.


You may also want to add:
```
  DATA_LOADER.NUM_WORKERS 0 \
  NUM_GPUS 2 \
  TRAIN.BATCH_SIZE 16 \
```

If you want to launch a quick job for debugging on your local machine.

## Resume from an Existing Checkpoint
If your checkpoint is trained by PyTorch, then you can add the following line in the command line, or you can also add it in the YAML config:

```
TRAIN.CHECKPOINT_FILE_PATH path_to_your_PyTorch_checkpoint
```

If the checkpoint in trained by Caffe2, then you can do the following:

```
TRAIN.CHECKPOINT_FILE_PATH path_to_your_Caffe2_checkpoint \
TRAIN.CHECKPOINT_TYPE caffe2
```

If you need to performance inflation on the checkpoint, remember to set `TRAIN.CHECKPOINT_INFLATE` to True.


## Perform Test
We have `TRAIN.ENABLE` and `TEST.ENABLE` to control whether training or testing is required for the current job. If only testing is preferred, you can set the `TRAIN.ENABLE` to False, and do not forget to pass the path to the model you want to test to TEST.CHECKPOINT_FILE_PATH.
```
python tools/run_net.py \
  --cfg configs/Kinetics/C2D_8x8_R50.yaml \
  DATA.PATH_TO_DATA_DIR path_to_your_dataset \
  TEST.CHECKPOINT_FILE_PATH path_to_your_checkpoint \
  TRAIN.ENABLE False \
```
## Run the Demo on Videos/Camera
### Classification
modify a `<pretrained_model_config_file>.yaml` in `configs/Kinetics/c2/` corresponding to the pretrained model you want to use and set the following settings (you can look at `demo/Kinetics/SLOWFAST_8x8_R50.yaml` for reference):
* `TRAIN.ENABLE: False`
* `TEST.ENABLE: False`
* `CHECKPOINT_TYPE: caffe2`
* `CHECKPOINT_FILE_PATH: "path/to/the/pre-trained/model.pkl"` (skip this if you decide to place the model in `OUTPUT_DIR` which is default to `./checkpoints/`)
* `DEMO.ENABLE: True`
* `DEMO.LABEL_FILE_PATH`: path to a CSV label file (already in `demo/Kinetics/kinetics_400_labels.csv`)..
* `DEMO.DATA_SOURCE`: Index to the camera source (if you want to run on a video, put the video path here and set values of `DEMO.DATA_SOURCE` in `slowfast/config/default.py` to `""`)
* `NUM_GPUS: 1` (only if running on a single CPU)
* `NUM_SHARDS: 1` (only if running on a single machine)
  
Optional:
* `DISPLAY_WIDTH`: custom display window width
* `DISPLAY_HEIGHT`: custom display window height
### Detection
modify a `<pretrained_model_config_file>.yaml` in `configs/AVA/c2/` corresponding to the pretrained model you want to use and set the following settings (you can look at `demo/AVA/SLOWFAST_32x2_R101_50_50.yaml` for reference):
* `TRAIN.ENABLE: False`
* `TEST.ENABLE: False`
* `DETECTION.ENABLE: True`
* `CHECKPOINT_TYPE: caffe2`
* `CHECKPOINT_FILE_PATH: "path/to/the/pre-trained/model.pkl"` (skip this if you decide to place the model in `OUTPUT_DIR` which is default to `./checkpoints/`)
* `DEMO.ENABLE: True`
* `DEMO.LABEL_FILE_PATH`: path to a text label file (already in `demo/AVA/ava.names`).
* `DEMO.DATA_SOURCE`: Index to the camera source (if you want to run on a video, put the video path here and set values of `DEMO.DATA_SOURCE` in `slowfast/config/default.py` to `""`)
* `NUM_GPUS: 1` (only if running on a single CPU)
* `NUM_SHARDS: 1` (only if running on a single machine)
  
Optional:
* `DEMO.DISPLAY_WIDTH`: custom display window width
* `DEMO.DISPLAY_HEIGHT`: custom display window height

### Run command
```
python \tools\run_net.py --cfg path/to/<pretrained_model_config_file>.yaml
```