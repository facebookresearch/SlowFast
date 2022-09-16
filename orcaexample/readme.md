# Orca PyTorch video object detection example on tiny-kinectic-400 dataset

We demonstrate how to easily run synchronous distributed Pytorch training using Pytorch Estimator of Project Orca in Bigdl. This is an example using [SlowFast](https://github.com/facebookresearch/SlowFast/tree/main/slowfast) to train on [ting-kinetic-400](https://github.com/Tramac/tiny-kinetics-400), a collection of video clips that cover 400 human action classes. We provide `bigdl` distributed PyTorch training backends for this example. You can run with either backend as you wish.

## Prepare the environment

We strongly recommend you install slowfast **Requirements** according to our slowfast document [here](https://github.com/analytics-zoo/SlowFast/blob/main/INSTALL.md#requirements). You don't need to build or install slowfast itself for now.
**Note**:
1. You do **not** need to install pytorch from source like [FAIR suggested](https://github.com/facebookresearch/SlowFast/blob/main/INSTALL.md#pytorch).
2. PIL is renamed to [Pillow](https://pypi.org/project/Pillow/), but their document hasn't been updated.
3. Their **torchvision** video decode backend are disabled for now, using pyav backend.[issue#181](https://github.com/facebookresearch/SlowFast/issues/181)

Clone **our** slowfast repo:
```
git clone https://github.com/analytics-zoo/SlowFast.git
```

## Prepare Dataset
This example only shows setup pipeline, so instead of using Kinetics(135GB), there is a public dataset much smaller here named tiny-kinetics-400(600MB), here goes [links](https://github.com/Tramac/tiny-kinetics-400), and we also provide a even smaller dataset called [tiny-kinetics-4](https://github.com/leonardozcm/SlowFast/releases/tag/tinykinectic400)(6MB) with only 4 classes. You can choose the one you like:
1. tiny-kinetics-4:
```
cd SlowFast/orcaexample
mkdir dataset
cd dataset
wget https://github.com/leonardozcm/SlowFast/releases/download/tinykinectic400/tiny-kinetics-400.zip
unzip tiny-kinetics-400.zip
```
done, you could see tiny-kinetics-400 folder and `tiny-kinetics-400.zip` under dataset/

2. tiny-kinetics-400:
```
cd SlowFast/orcaexample
mkdir dataset
cd dataset
git clone https://github.com/Tramac/tiny-kinetics-400.git
```
And download dataset archive [link](https://github.com/leonardozcm/SlowFast/releases/download/tinykinectic400/tiny-kinetics-400.zip0), extract it and place them under [here](https://github.com/Tramac/tiny-kinetics-400/tree/main/data)
more information refer to [document](https://github.com/Tramac/tiny-kinetics-400). And run gen_csv.py to generate train.csv and val.csv under `orcaexample/dataset/tiny-kinetics-400/data`

```
cd SlowFast/orcaexample
python gen_csv.py
```
train.csv be like:
```
dataset/tiny-kinetics-400/data/abseiling/-3B32lodo2M_000059_000069.mp4 0
dataset/tiny-kinetics-400/data/abseiling/_4YTwq0-73Y_000044_000054.mp4 0
dataset/tiny-kinetics-400/data/abseiling/_EDt9CNqqxk_000260_000270.mp4 0
...
```

Note that for yarn mode, you need to pack datasets/ yourself.

```
zip -q -r tiny-kinetics-400.zip tiny-kinetics-400
```

## Install slowfast
Please **make sure** your slowfast works--After having the above dependencies, run:
```
cd SlowFast
python setup.py build develop
```
test slowfast:
```
cd SlowFast
python tools/run_net.py --cfg ./C2D_8x8_R50.yaml --opts NUM_GPUS 0 TRAIN.BATCH_SIZE 8 SOLVER.BASE_LR 0.0125 DATA.PATH_TO_DATA_DIR ./dataset/tiny-kinetics-400/data
```
Set NUM_GPUS=0 makes it run on cpu. 

Build and Install PySlowFast
```
pip uninstall slowfast -y
python setup.py build install
```
you will see slowfast in your site-packages
```
Using /home/usrname/anaconda3/envs/py37/lib/python3.7/site-packages
Finished processing dependencies for slowfast==1.0
```

## Run example
You can run this example on local mode (default) and yarn-client mode.

- Run with Spark Local mode:
```bash
python kinectic.py --cluster_mode local
```

- Run with Yarn-Client mode:
```bash
python kinectic.py --cluster_mode yarn
```

You can run this example with bigdl backend (default), ray backend, or spark backend. 

- Run with bigdl backend:
```bash
python kinectic.py --backend bigdl
```

**Options**
* `--cluster_mode` The mode of spark cluster. Either "local" or "yarn". Default is "local".
* `--backend` The backend of PyTorch Estimator. Either "bigdl", "ray" or "spark. Default is "bigdl".
* `--cfg_files` The path to config file
* `--executor_memory` executor_memory, default to `5g`
* `--driver_memory` driver memory, default to `5g`
* `--opts` See slowfast/config/defaults.py for all options


## Results

**For "bigdl" backend**

You can find the training results as:
```
22-09-16 10:04:31 [Thread-3] INFO  DistriOptimizer$:432 - [Epoch 2 4/8][Iteration 3][Wall Clock 206.8350893s] Trained 4.0 records in 73.3313766 seconds. Throughput is 0.05454691 records/second. Loss is 29.229464. 
22-09-16 10:05:25 [Thread-3] INFO  DistriOptimizer$:432 - [Epoch 2 8/8][Iteration 4][Wall Clock 261.11777s] Trained 4.0 records in 54.2826807 seconds. Throughput is 0.07368833 records/second. Loss is 29.40612. 
22-09-16 10:05:25 [Thread-3] INFO  DistriOptimizer$:474 - [Epoch 2 8/8][Iteration 4][Wall Clock 261.11777s] Epoch finished. Wall clock time is 282940.3175 ms
22-09-16 10:05:25 [Thread-3] INFO  DistriOptimizer$:112 - [Epoch 2 8/8][Iteration 4][Wall Clock 261.11777s] Validate model...
22-09-16 10:05:43 [Thread-3] INFO  DistriOptimizer$:181 - [Epoch 2 8/8][Iteration 4][Wall Clock 261.11777s] validate model throughput is 0.21665977 records/second
22-09-16 10:05:43 [Thread-3] INFO  DistriOptimizer$:184 - [Epoch 2 8/8][Iteration 4][Wall Clock 261.11777s] Top1Accuracy is Accuracy(correct: 1, count: 4, accuracy: 0.25)
```
