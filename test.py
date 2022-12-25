import base64
import json

import cv2
import numpy as np
import six
from matplotlib.colors import hsv_to_rgb, rgb_to_hsv
from numpy import cos, pi, sin
from PIL import Image


def base64_to_PIL(string):
    try:
        base64_data = base64.b64decode(string)
        buf = six.BytesIO()
        buf.write(base64_data)
        buf.seek(0)
        img = Image.open(buf).convert("RGB")
        return img
    except:
        return None


def read_json(p):
    with open(p) as f:
        jsonData = json.loads(f.read())
    shapes = jsonData.get("shapes")
    imageData = jsonData.get("imageData")
    lines = []
    labels = []
    for shape in shapes:
        lines.append(shape["points"])
        [x0, y0], [x1, y1] = shape["points"]
        label = shape["label"]
        if label == "0":
            if abs(y1 - y0) > 500:
                label = "1"
        elif label == "1":
            if abs(x1 - x0) > 500:
                label = "0"

        labels.append(label)
    img = base64_to_PIL(imageData)
    return img, lines, labels


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


from data.preprocesing import img_argument

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    path = "./datasets/train_samples/5.json"
    img, lines, labels = read_json(path)
    plt.imshow(img)
    plt.show()
    print(img)
    print(lines)
    print(labels)

    img = plot_lines(img, lines)
    plt.imshow(img)
    plt.show()

    img = fill_lines(img, lines)
    plt.imshow(img)
    plt.show()

    img = Image.open("./datasets/train_samples/5.jpg")
    img, newlines, newlables = img_argument(img, lines, labels, size=(1024, 1024))
    plt.imshow(img)
    plt.show()
    print(newlables)
    print(newlines)
