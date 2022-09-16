# Installation

## Requirements
- Python >= 3.8
- Numpy
- PyTorch >= 1.3
- [fvcore](https://github.com/facebookresearch/fvcore/): `pip install 'git+https://github.com/facebookresearch/fvcore'`
- [torchvision](https://github.com/pytorch/vision/) that matches the PyTorch installation.
  You can install them together at [pytorch.org](https://pytorch.org) to make sure of this.
- simplejson: `pip install simplejson`
- GCC >= 4.9
- PyAV: `conda install av -c conda-forge`
- ffmpeg (4.0 is prefereed, will be installed along with PyAV)
- PyYaml: (will be installed along with fvcore)
- tqdm: (will be installed along with fvcore)
- iopath: `pip install -U iopath` or `conda install -c iopath iopath`
- psutil: `pip install psutil`
- OpenCV: `pip install opencv-python`
- torchvision: `pip install torchvision` or `conda install torchvision -c pytorch`
- tensorboard: `pip install tensorboard`
- moviepy: (optional, for visualizing video on tensorboard) `conda install -c conda-forge moviepy` or `pip install moviepy`
- PyTorchVideo: `pip install pytorchvideo`
- [Detectron2](https://github.com/facebookresearch/detectron2):
- FairScale: `pip install 'git+https://github.com/facebookresearch/fairscale'`
```
    pip install -U torch torchvision cython
    pip install -U 'git+https://github.com/facebookresearch/fvcore.git' 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
    git clone https://github.com/facebookresearch/detectron2 detectron2_repo
    pip install -e detectron2_repo
    # You can find more details at https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md
```

## PySlowFast

Clone the PySlowFast Video Understanding repository.
```
git clone https://github.com/facebookresearch/slowfast
```

### Test PySlowFast

After having the above dependencies, run:
```
git clone https://github.com/facebookresearch/slowfast
cd SlowFast
python setup.py build develop
```

Now the installation is finished, run the pipeline with:
```
python tools/run_net.py --cfg ./C2D_8x8_R50.yaml --opts NUM_GPUS 0 TRAIN.BATCH_SIZE 8 SOLVER.BASE_LR 0.0125 DATA.PATH_TO_DATA_DIR ./dataset/tiny-kinetics-400/data
```
### Build and Install PySlowFast
```
pip uninstall slowfast -y
python setup.py build install
```
you will see slowfast in your site-packages
```
Using /home/usrname/anaconda3/envs/py37/lib/python3.7/site-packages
Finished processing dependencies for slowfast==1.0
```
