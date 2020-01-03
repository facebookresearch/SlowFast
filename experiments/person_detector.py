import os
import sys
import cv2 as cv
import numpy as np
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.video_visualizer import VideoVisualizer
from detectron2.data import MetadataCatalog
from detectron2.utils.logger import setup_logger
setup_logger()

def main(args):
    cfg_file = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(cfg_file))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = .5
    cfg.MODEL.WEIGHTS = "detectron2://COCO-Detection/faster_rcnn_R_50_FPN_3x/137849458/model_final_280758.pkl"
    predictor = DefaultPredictor(cfg)
    cap = cv.VideoCapture(-1)
    if not cap.isOpened():
        raise RuntimeError('Camera not found!')

    while True:
        _, frame = cap.read()
        outputs = predictor(frame)
        print(type(outputs))
        print(outputs)
        sys.exit()


if __name__ == "__main__":
    main(sys.argv[1:])

