import cv2
import numpy as np
from numpy import cos, pi, sin
from PIL import Image


def plot_lines(img, lines, linetype=2):
    tmp = np.copy(img)
    for line in lines:
        p1, p2 = line
        cv2.line(
            tmp,
            (int(p1[0]), int(p1[1])),
            (int(p2[0]), int(p2[1])),
            (0, 0, 0),
            linetype,
            lineType=cv2.LINE_AA,
        )

    return Image.fromarray(tmp)


def fill_lines(img, lines, linetype=2):
    tmp = np.copy(img)
    for line in lines:
        p1, p2 = line
        cv2.line(
            tmp,
            (int(p1[0]), int(p1[1])),
            (int(p2[0]), int(p2[1])),
            255,
            linetype,
            lineType=cv2.LINE_AA,
        )

    return tmp


def angle_transpose(p, angle, w, h):
    x, y = p
    if angle == 90:
        x, y = y, w - x
    elif angle == 180:
        x, y = w - x, h - y
    elif angle == 270:
        x, y = h - y, x
    return x, y


def rotate(x, y, angle, cx, cy):
    angle = angle * pi / 180
    x_new = (x - cx) * cos(angle) - (y - cy) * sin(angle) + cx
    y_new = (x - cx) * sin(angle) + (y - cy) * cos(angle) + cy
    return x_new, y_new


def img_argument(img, lines, labels, size=(512, 512)):
    w, h = img.size
    if np.random.randint(0, 100) > 80:
        degree = np.random.uniform(-5, 5)
    else:
        degree = 0
    # degree = np.random.uniform(-5,5)
    newlines = []
    for line in lines:
        p1, p2 = line
        p1 = rotate(p1[0], p1[1], degree, w / 2, h / 2)
        p2 = rotate(p2[0], p2[1], degree, w / 2, h / 2)
        newlines.append([p1, p2])
    # img = img.rotate(-degree,center=(w/2,h/2),resample=Image.BILINEAR,fillcolor=(128,128,128))
    img = img.rotate(-degree, center=(w / 2, h / 2), resample=Image.BILINEAR)
    angle = np.random.choice([0, 90, 180, 270], 1)[0]
    newlables = []
    for i in range(len(newlines)):
        p1, p2 = newlines[i]
        p1 = angle_transpose(p1, angle, w, h)
        p2 = angle_transpose(p2, angle, w, h)
        newlines[i] = [p1, p2]
        if angle in [90, 270]:
            if labels[i] == "0":
                newlables.append("1")
            else:
                newlables.append("0")
        else:
            newlables.append(labels[i])

    if angle == 90:
        img = img.transpose(Image.ROTATE_90)
    elif angle == 180:
        img = img.transpose(Image.ROTATE_180)
    elif angle == 270:
        img = img.transpose(Image.ROTATE_270)

    return img, newlines, newlables
