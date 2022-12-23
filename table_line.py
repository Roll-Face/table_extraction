"""
@author: nam157
"""
from typing import Any, Tuple

import cv2
import numpy as np
import tensorflow as tf
from skimage import measure

from utils import (adjust_lines, draw_lines, get_table_line, letterbox_image,
                   line_to_line, minAreaRectbox)


def detect_line(args, img: Any, size: Tuple[int], model: tf.Module):
    sizew, sizeh = size
    if isinstance(img, str):
        img = cv2.imread(img)
    else:
        img = img.copy()
    inputBlob, fx, fy = letterbox_image(img[..., ::-1], (sizew, sizeh))
    pred = model.predict(np.array([np.array(inputBlob) / 255.0]))
    pred = pred[0]
    vpred = pred[..., 1] > args.vprob
    hpred = pred[..., 0] > args.hprob
    vpred = vpred.astype(int)
    hpred = hpred.astype(int)

    colboxes = get_table_line(vpred, axis=1, lineW=args.col)
    rowboxes = get_table_line(hpred, axis=0, lineW=args.row)
    ccolbox = []
    crowlbox = []
    if len(rowboxes) > 0:
        rowboxes = np.array(rowboxes)
        rowboxes[:, [0, 2]] = rowboxes[:, [0, 2]] / fx
        rowboxes[:, [1, 3]] = rowboxes[:, [1, 3]] / fy
        xmin = rowboxes[:, [0, 2]].min()
        xmax = rowboxes[:, [0, 2]].max()
        ymin = rowboxes[:, [1, 3]].min()
        ymax = rowboxes[:, [1, 3]].max()
        ccolbox = [[xmin, ymin, xmin, ymax], [xmax, ymin, xmax, ymax]]
        rowboxes = rowboxes.tolist()

    if len(colboxes) > 0:
        colboxes = np.array(colboxes)
        colboxes[:, [0, 2]] = colboxes[:, [0, 2]] / fx
        colboxes[:, [1, 3]] = colboxes[:, [1, 3]] / fy

        xmin = colboxes[:, [0, 2]].min()
        xmax = colboxes[:, [0, 2]].max()
        ymin = colboxes[:, [1, 3]].min()
        ymax = colboxes[:, [1, 3]].max()
        colboxes = colboxes.tolist()
        crowlbox = [[xmin, ymin, xmax, ymin], [xmin, ymax, xmax, ymax]]
    rowboxes += crowlbox
    colboxes += ccolbox

    rboxes_row_, rboxes_col_ = adjust_lines(rowboxes, colboxes, alph=args.alph)
    rowboxes += rboxes_row_
    colboxes += rboxes_col_

    nrow = len(rowboxes)
    ncol = len(colboxes)
    for i in range(nrow):
        for j in range(ncol):
            rowboxes[i] = line_to_line(
                rowboxes[i], colboxes[j], args.alpha_line_to_line
            )
            colboxes[j] = line_to_line(
                colboxes[j], rowboxes[i], args.alpha_line_to_line
            )

    tmp = np.zeros(img.shape[:2], dtype="uint8")
    tmp = draw_lines(tmp, rowboxes + colboxes, color=255, lineW=4)
    labels = measure.label(tmp < 255, connectivity=2)
    regions = measure.regionprops(labels)
    ceilboxes = minAreaRectbox(regions, False, tmp.shape[0], tmp.shape[0], True, True)
    ceilboxes = np.array(ceilboxes)

    return ceilboxes
