from time import time

import numpy as np
import pandas as pd
import cv2
import torch

import slowfast.utils.checkpoint as cu
import slowfast.utils.distributed as du
import slowfast.utils.logging as logging
import slowfast.utils.misc as misc
from slowfast.datasets import loader
from slowfast.models import model_builder
# from slowfast.utils.meters import TestMeter

logger = logging.get_logger(__name__)

def imresize(im, dsize):
    '''
    Resize the image to the specified square sizes and 
    maintain the original aspect ratio using padding.

    Args:

        im -- input image.
        dsize -- output sizes, can be an integer or a tuple.

    Returns:

        resized image.
    '''
    if type(dsize) is int:
        dsize = (dsize, dsize)
        
    im_h, im_w, _ = im.shape
    to_w, to_h = dsize
    scale_ratio = min(to_w/im_w, to_h/im_h)
    new_im = cv2.resize(im,(0, 0), 
                        fx=scale_ratio, fy=scale_ratio, 
                        interpolation=cv2.INTER_AREA)
    new_h, new_w, _ = new_im.shape
    padded_im = np.full((to_h, to_w, 3), 128)
    x1 = (to_w-new_w)//2
    x2 = x1 + new_w
    y1 = (to_h-new_h)//2
    y2 = y1 + new_h
    padded_im[y1:y2, x1:x2, :] = new_im 
    
    return padded_im

class VideoReader(object):

    def __init__(self, source):
        self.source = source
        try:  # OpenCV needs int to read from webcam
            self.source = int(source)
        except ValueError:
            pass

    def __iter__(self):
        self.cap = cv2.VideoCapture(self.source)
        # self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        # self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        if not self.cap.isOpened():
            raise IOError('Video {} cannot be opened'.format(self.source))
        return self

    def __next__(self):
        was_read, frame = self.cap.read()
        if not was_read:
            # raise StopIteration
            ## reiterate the video instead of quiting.
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frame = None

        return was_read, frame

    def clean(self):
        self.cap.release()
        cv2.destroyAllWindows()


def predict(cfg):
    # Set random seed from configs.
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)

    # Setup logging format.
    logging.setup_logging()

    # Print config.
    logger.info("Predict with config:")
    logger.info(cfg)
    # Build the video model and print model statistics.
    model = model_builder.build_model(cfg)
    model.eval()
    misc.log_model_info(model)

   # Load a checkpoint to test if applicable.
    if cfg.TEST.CHECKPOINT_FILE_PATH != "":
        ckpt = cfg.TEST.CHECKPOINT_FILE_PATH
        convert_from_caffe2 = cfg.TEST.CHECKPOINT_TYPE == "caffe2"
    elif cu.has_checkpoint(cfg.OUTPUT_DIR):
        ckpt = cfg.OUTPUT_DIR
        convert_from_caffe2 = False
    elif cfg.TRAIN.CHECKPOINT_FILE_PATH != "":
        # If no checkpoint found in TEST.CHECKPOINT_FILE_PATH or in the current
        # checkpoint folder, try to load checkpint from
        # TRAIN.CHECKPOINT_FILE_PATH and test it.
        ckpt = cfg.TRAIN.CHECKPOINT_FILE_PATH
        convert_from_caffe2 = cfg.TRAIN.CHECKPOINT_TYPE == "caffe2"
    else:
        raise NotImplementedError("Unknown way to load checkpoint.")

    cu.load_checkpoint(
        ckpt,
        model,
        cfg.NUM_GPUS > 1,
        None,
        inflation=False,
        convert_from_caffe2=convert_from_caffe2,
    )

    # TODO: Take inputs from camera proceed below, definitely use for loop.
    # Load the labels of Kinectics-400 dataset
    labels_df = pd.read_csv(cfg.PREDICT.LABEL_FILE_PATH)
    labels = labels_df['name'].values
    img_provider = VideoReader(cfg.PREDICT.SOURCE)
    frames = []
    label = 'predicting...'
    s = 0.
    i = -1
    for able_to_read, frame in img_provider:
        i += 1
        if not able_to_read:
            # when reaches the end frame, clear the buffer and continue to the next one.
            frames = []
            continue

        if len(frames) != cfg.DATA.NUM_FRAMES*cfg.DATA.SAMPLING_RATE:
            frame_processed = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_processed = imresize(frame_processed, 256)
            frames.append(frame_processed)
            
        if len(frames) == cfg.DATA.NUM_FRAMES*cfg.DATA.SAMPLING_RATE:
            start = time()
            # Perform color normalization.
            inputs = torch.tensor(frames).float()
            inputs = inputs / 255.0
            inputs = inputs - torch.tensor(cfg.DATA.MEAN)
            inputs = inputs / torch.tensor(cfg.DATA.STD)
            # T H W C -> C T H W.
            inputs = inputs.permute(3, 0, 1, 2)
            # 1 C T H W.
            inputs = inputs[None, :, :, :, :]
            # Sample frames for the fast pathway.
            index = torch.linspace(0, inputs.shape[2] - 1, cfg.DATA.NUM_FRAMES).long()
            fast_pathway = torch.index_select(inputs, 2, index)
            logger.info('fast_pathway.shape={}'.format(fast_pathway.shape))
            # Sample frames for the slow pathway.
            index = torch.linspace(0, fast_pathway.shape[2] - 1, 
                                fast_pathway.shape[2]//cfg.SLOWFAST.ALPHA).long()
            slow_pathway = torch.index_select(fast_pathway, 2, index)
            logger.info('slow_pathway.shape={}'.format(slow_pathway.shape))
            inputs = [slow_pathway, fast_pathway]
            # Transfer the data to the current GPU device.
            if isinstance(inputs, (list,)):
                for i in range(len(inputs)):
                    inputs[i] = inputs[i].cuda(non_blocking=True)
            else:
                inputs = inputs.cuda(non_blocking=True)

            # Perform the forward pass.
            preds = model(inputs)
            # Gather all the predictions across all the devices to perform ensemble.
            if cfg.NUM_GPUS > 1:
                preds = du.all_gather(preds)[0]

            logger.info(preds.shape)
            ## Option 1: single label inference selected from the highest probability entry.
            label_idx = torch.argmax(preds, -1).cpu()
            logger.info(label_idx)
            label = labels[label_idx]
            ## Option 2: multi-label inferencing selected from probability entries > threshold
            

            # remove the oldest frame in the buffer to make place for the new one.
            # frames.pop(0)
            frames = []
            s = time() - start

        # Display predicted label to frame
        cv2.putText(frame, 'Action: {}'.format(label), (20, 30), 
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1, color=(0, 235, 0), thickness=3)
        # Display prediction speed to frame
        cv2.putText(frame, 'Speed: {:.2f}s'.format(s), (20, 60), 
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1, color=(0, 235, 0), thickness=3)
        # Display the frame
        cv2.imshow('SlowFast', frame)   
        # hit Esc to quit the demo.
        key = cv2.waitKey(1)
        if key == 27:
            break

    img_provider.clean()
