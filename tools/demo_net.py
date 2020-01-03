from time import time

import numpy as np
import pandas as pd
import cv2
import torch

import slowfast.utils.checkpoint as cu
import slowfast.utils.distributed as du
import slowfast.utils.logging as logging
import slowfast.utils.misc as misc
from slowfast.models import model_builder
from slowfast.datasets.cv2_transform import scale

logger = logging.get_logger(__name__)


class VideoReader(object):

    def __init__(self, cfg):
        self.source = cfg.DEMO.DATA_SOURCE
        self.width = cfg.DEMO.DISPLAY_WIDTH
        self.height = cfg.DEMO.DISPLAY_HEIGHT
        try:  # OpenCV needs int to read from webcam
            self.source = int(self.source)
        except ValueError:
            pass

    def __iter__(self):
        self.cap = cv2.VideoCapture(self.source)
        if self.width > 0 and self.height > 0:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
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


def demo(cfg):
    # Set random seed from configs.
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)

    # Setup logging format.
    logging.setup_logging()

    # Print config.
    logger.info("Run demo with config:")
    logger.info(cfg)
    # Build the video model and print model statistics.
    model = model_builder.build_model(cfg)
    model.eval()
    misc.log_model_info(model)

   # Load a checkpoint to test if applicable.
    if cfg.TEST.CHECKPOINT_FILE_PATH != "":
        ckpt = cfg.TEST.CHECKPOINT_FILE_PATH
    elif cu.has_checkpoint(cfg.OUTPUT_DIR):
        ckpt = cu.get_last_checkpoint(cfg.OUTPUT_DIR)
    elif cfg.TRAIN.CHECKPOINT_FILE_PATH != "":
        # If no checkpoint found in TEST.CHECKPOINT_FILE_PATH or in the current
        # checkpoint folder, try to load checkpoint from
        # TRAIN.CHECKPOINT_FILE_PATH and test it.
        ckpt = cfg.TRAIN.CHECKPOINT_FILE_PATH
    else:
        raise NotImplementedError("Unknown way to load checkpoint.")

    cu.load_checkpoint(
        ckpt,
        model,
        cfg.NUM_GPUS > 1,
        None,
        inflation=False,
        convert_from_caffe2= "caffe2" in [cfg.TEST.CHECKPOINT_TYPE, cfg.TRAIN.CHECKPOINT_TYPE],
    )

    # Load the labels of Kinectics-400 dataset
    labels_df = pd.read_csv(cfg.DEMO.LABEL_FILE_PATH)
    labels = labels_df['name'].values
    img_provider = VideoReader(cfg)
    frames = []
    # # Option 1
    # pred_label = ''
    # Option 2  
    pred_labels = []
    s = 0.
    for able_to_read, frame in img_provider:
        if not able_to_read:
            # when reaches the end frame, clear the buffer and continue to the next one.
            frames = []
            continue

        if len(frames) != cfg.DATA.NUM_FRAMES*cfg.DATA.SAMPLING_RATE:
            frame_processed = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_processed = scale(256, frame_processed)
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

            ## Option 1: single label inference selected from the highest probability entry.
            # label_id = preds.argmax(-1).cpu()
            # pred_label = labels[label_id]
            # Option 2: multi-label inferencing selected from probability entries > threshold
            label_ids = torch.nonzero(preds.squeeze() > .1).reshape(-1).cpu().detach().numpy()
            pred_labels = labels[label_ids]
            logger.info(pred_labels)
            if not list(pred_labels):
                pred_labels = ['Unknown']

            # remove the oldest frame in the buffer to make place for the new one.
            # frames.pop(0)
            frames = []
            s = time() - start

        # #************************************************************
        # # Option 1
        # #************************************************************
        # # Display prediction speed to frame
        # cv2.putText(frame, 'Speed: {:.2f}s'.format(s), (20, 30), 
        #             fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        #             fontScale=1, color=(0, 235, 0), thickness=3)
        # # Display predicted label to frame.
        # cv2.putText(frame, 'Action: {}'.format(pred_label), (20, 60), 
        #             fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        #             fontScale=1, color=(0, 235, 0), thickness=3)
        #************************************************************
        # Option 2
        #************************************************************
        # Display prediction speed to frame
        cv2.putText(frame, 'Speed: {:.2f}s'.format(s), (20, 30), 
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1, color=(0, 235, 0), thickness=3)
        # Display predicted labels to frame.
        y_offset = 60
        cv2.putText(frame, 'Action:', (20, y_offset), 
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=1, color=(0, 235, 0), thickness=3)        
        for pred_label in pred_labels:
            y_offset += 30
            cv2.putText(frame, '{}'.format(pred_label), (20, y_offset), 
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=1, color=(0, 235, 0), thickness=3)

        # Display the frame
        cv2.imshow('SlowFast', frame)   
        # hit Esc to quit the demo.
        key = cv2.waitKey(1)
        if key == 27:
            break

    img_provider.clean()
