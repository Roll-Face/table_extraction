import base64
import json

import cv2
import numpy as np
import six
from matplotlib.colors import hsv_to_rgb, rgb_to_hsv
from numpy import cos, pi, sin
from PIL import Image
from plot_line import fill_lines, img_argument


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


def img_resize(im, lines, target_size=600, max_size=1500):
    w, h = im.size
    im_size_min = np.min(im.size)
    im_size_max = np.max(im.size)

    im_scale = float(target_size) / float(im_size_min)
    if max_size is not None:
        if np.round(im_scale * im_size_max) > max_size:
            im_scale = float(max_size) / float(im_size_max)

    im = im.resize((int(w * im_scale), int(h * im_scale)), Image.BICUBIC)
    N = len(lines)
    for i in range(N):
        p1, p2 = lines[i]
        p1 = p1[0] * im_scale, p1[1] * im_scale
        p2 = p2[0] * im_scale, p2[1] * im_scale
        lines[i] = [p1, p2]

    return im, lines


def get_img_label(p, size, linetype=1):
    img, lines, labels = read_json(p)
    img, lines = img_resize(img, lines, target_size=512, max_size=1024)
    img, lines, labels = img_argument(img, lines, labels, size)
    img, lines, labels = get_random_data(img, lines, labels, size=size)

    lines = np.array(lines)
    labels = np.array(labels)
    labelImg0 = np.zeros(size[::-1], dtype="uint8")
    labelImg1 = np.zeros(size[::-1], dtype="uint8")

    ind = np.where(labels == "0")[0]
    labelImg0 = fill_lines(labelImg0, lines[ind], linetype=linetype)
    ind = np.where(labels == "1")[0]
    labelImg1 = fill_lines(labelImg1, lines[ind], linetype=linetype)

    labelY = np.zeros((size[1], size[0], 2), dtype="uint8")
    labelY[:, :, 0] = labelImg0
    labelY[:, :, 1] = labelImg1
    labelY = labelY > 0
    return np.array(img), lines, labelY


def rand(a=0, b=1):
    return np.random.rand() * (b - a) + a


def get_random_data(
    image, lines, labels, size=(1024, 1024), jitter=0.3, hue=0.1, sat=1.5, val=1.5
):
    """random preprocessing for real-time data augmentation"""

    iw, ih = image.size

    # resize image
    w, h = size
    new_ar = w / h * rand(1 - jitter, 1 + jitter) / rand(1 - jitter, 1 + jitter)
    # scale = rand(.2, 2)
    scale = rand(0.2, 3)
    if new_ar < 1:
        nh = int(scale * h)
        nw = int(nh * new_ar)
    else:
        nw = int(scale * w)
        nh = int(nw / new_ar)
    image = image.resize((nw, nh), Image.BICUBIC)

    # place image
    dx = int(rand(0, w - nw))
    dy = int(rand(0, h - nh))
    new_image = Image.new("RGB", (w, h), (128, 128, 128))
    new_image.paste(image, (dx, dy))
    image = new_image

    # distort image
    hue = rand(-hue, hue)
    sat = rand(1, sat) if rand() < 0.5 else 1 / rand(1, sat)
    val = rand(1, val) if rand() < 0.5 else 1 / rand(1, val)
    x = rgb_to_hsv(np.array(image) / 255.0)
    x[..., 0] += hue
    x[..., 0][x[..., 0] > 1] -= 1
    x[..., 0][x[..., 0] < 0] += 1
    x[..., 1] *= sat
    x[..., 2] *= val
    x[x > 1] = 1
    x[x < 0] = 0
    image_data = hsv_to_rgb(x)  # numpy array, 0 to 1
    N = len(lines)
    for i in range(N):
        p1, p2 = lines[i]
        p1 = p1[0] * nw / iw + dx, p1[1] * nh / ih + dy
        p2 = p2[0] * nw / iw + dx, p2[1] * nh / ih + dy
        lines[i] = [p1, p2]
    return image_data, lines, labels


def gen(paths, batchsize=2, linetype=2):
    num = len(paths)
    i = 0
    while True:
        # sizes = [512,512,512,512,640,1024] ##多尺度训练
        # size = np.random.choice(sizes,1)[0]
        size = 640

        X = np.zeros((batchsize, size, size, 3))
        Y = np.zeros((batchsize, size, size, 2))
        for j in range(batchsize):
            if i >= num:
                i = 0
                np.random.shuffle(paths)
            p = paths[i]
            i += 1

            # linetype=2
            img, lines, labelImg = get_img_label(
                p, size=(size, size), linetype=linetype
            )
            X[j] = img
            Y[j] = labelImg

        yield X, Y
