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
    visualizer = VideoVisualizer(MetadataCatalog.get(cfg.DATASETS.TRAIN[0]))
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError('Camera not found!')

    while cap.isOpened():
        _, frame = cap.read()
        outputs = predictor(frame)
        frame_out = visualizer.draw_instance_predictions(frame[:, :, ::-1], outputs["instances"].to("cpu"))
        cv.imshow('Detectron2: Object Detection', frame_out.get_image()[:, :, ::-1])
        key = cv.waitKey(1)
        if key == 27:
            break



if __name__ == "__main__":
    main(sys.argv[1:])

