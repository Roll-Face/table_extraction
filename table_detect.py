"""
@author: nam157
"""

import cv2
import numpy as np
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor


def init_detectron2(file_config: str, file_checkpoint: str):
    cfg = get_cfg()
    cfg.merge_from_file(file_config)
    cfg.MODEL.WEIGHTS = file_checkpoint
    predictor = DefaultPredictor(cfg)
    return predictor


def make_prediction_header(img, predictor):
    outputs = predictor(img)
    for i, box in enumerate(
        outputs["instances"].get_fields()["pred_boxes"].tensor.to("cpu").numpy()
    ):
        x1, y1, x2, y2 = box
        y2 += 6
        header = np.array(img[int(0) : int(y1), int(0) : int(x2)], copy=True)
    return header


def make_prediction(img, predictor):

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
    border = cv2.copyMakeBorder(img, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=[255, 255])
    outputs = predictor(border)

    table_list = []
    table_coords = []
    for i, box in enumerate(
        outputs["instances"].get_fields()["pred_boxes"].tensor.to("cpu").numpy()
    ):
        x1, y1, x2, y2 = box
        table_list.append(
            np.array(img[int(y1) - 10 : int(y2) + 30, int(x1) : int(x2)], copy=True)
        )
        table_coords.append([int(x1), int(y1), int(x2 - x1), int(y2 - y1)])
    return table_list, table_coords
