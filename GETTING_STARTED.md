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

Currently, demo on multiple GPUs is not supported. Set the following in your config file:
* `NUM_GPUS: 1`
* `NUM_SHARDS: 1`
* `DEMO.WEBCAM`: Set this to an index of a camera for demo using webcam. Otherwise, set `DEMO.INPUT_VIDEO` to path of an input video.
* `DEMO.ENABLE: True`
* `DEMO.LABEL_FILE_PATH`: path to a json label file that map {label: label_id}.
* `CHECKPOINT_FILE_PATH: "path/to/the/pre-trained/model.pkl"` (skip this if you decide to place the model in `OUTPUT_DIR` which is default to `./checkpoints/`)

Optional:
* `DEMO.DISPLAY_WIDTH`: custom display window width.
* `DEMO.DISPLAY_HEIGHT`: custom display window height.
* `DEMO.OUTPUT_FILE`: set this as a path if you want to write outputs to a video file instead of displaying it in a window.
* `DEMO.BUFFER_SIZE`: overlapping number of frames between 2 consecutive input clips. Set this to a positive number to make more frequent predictions
(at the expense of slower speed).

If you want to only run the demo process, set `TRAIN.ENABLE` and `TEST.ENABLE` to False.

### Classification
Modify a `<pretrained_model_config_file>.yaml` in `configs/Kinetics/` corresponding to the pretrained model you want to use (you can look at `demo/Kinetics/SLOWFAST_8x8_R50.yaml` for reference).

### Detection
Modify a `<pretrained_model_config_file>.yaml` in `configs/AVA/` corresponding to the pretrained model you want to use (you can look at `demo/AVA/SLOWFAST_32x2_R101_50_50.yaml` for reference)

Optional:
* `DEMO.DETECTRON2_THRESH`: Set a threshold for choosing bouding boxes outputed by Detectron2. (Default to 0.9)
* Pick a different [Detectron2](https://github.com/facebookresearch/detectron2/blob/master/MODEL_ZOO.md) Object Detection model config and weights. Set the parameters: `DEMO.DETECTRON2_CFG` and `DEMO.DETECTRON2_WEIGHTS`. (Default to `faster_rcnn_R_50_FPN_3x.yaml` config and the corresponding weights)

### Run command
```
python \tools\run_net.py --cfg path/to/<pretrained_model_config_file>.yaml
```
