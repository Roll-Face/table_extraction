import cv2
import easyocr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from models import table_net
from table_detect import init_detectron2, make_prediction
from table_line import detect_line
from utils import coordinates_img, draw_boxes_v2


def inference(args):
    # Load model
    model = table_net((None, None, 3), 2)
    model.load_weights(args.model_table_line)
    detectron2 = init_detectron2(args.config_detectron2, args.model_detectron2)
    reader_ocr = easyocr.Reader(["en"])

    # Load image
    image = cv2.imread(args.input_path)
    table_list, table_coords = make_prediction(image, detectron2)
    img = table_list[0]
    im = img.copy()

    # Detect line
    ceilboxes = detect_line(args=args, img=img, size=args.size, model=model)

    # Sort bbox
    ls = coordinates_img(ceilboxes)
    ls.sort(key=lambda x: x[1])

    # Table ocr
    outer = []
    for i in ls:
        top_left_x, top_left_y, bot_right_x, bot_right_y = i
        sub_img = im[top_left_y:bot_right_y, top_left_x:bot_right_x]
        h, w, c = sub_img.shape
        if w > 20:
            results = reader_ocr.readtext(sub_img)
            plain_text = ""
            for box in results:
                plain_text += box[-2]
                plain_text += " "
            print(plain_text)
            outer.append(plain_text)

    # Save file
    arr = np.array(outer)
    dataframe = pd.DataFrame(arr.reshape(len(arr) // 5, 5))
    dataframe.to_csv("./results/demo.csv", index=False)


if __name__ == "__main__":
    import yaml

    from utils import AttrDict

    with open("./base_config.yaml", "r", encoding="utf8") as f:
        opt = yaml.safe_load(f)
    opt = AttrDict(opt)

    inference(opt)
