import argparse
import os
from glob import glob

import cv2
import matplotlib.pyplot as plt
import yaml
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

from models import table_net
from preprocesing import gen
from table_detect import init_detectron2, make_prediction
from table_line import detect_line
from utils import AttrDict, draw_boxes_v2


def train(args):

    trainP, testP = train_test_split(
        glob(args.file_label + "/*.json"), test_size=0.1, random_sate=121
    )
    trainloader = gen(trainP, batchsize=args.batchsize, linetype=1)
    testloader = gen(testP, batchsize=args.batchsize, linetype=1)
    os.makedirs(args.save_checkpoint, exist_ok=True)
    checkpointer = ModelCheckpoint(
        filepath=args.save_checkpoint,
        monitor="loss",
        verbose=0,
        save_weights_only=True,
        save_best_only=True,
    )
    scheduler = ReduceLROnPlateau(
        monitor=args.monitor,
        factor=args.factor,
        patience=args.patience,
        mode=args.mode,
        min_lr=args.min_lr,
    )
    model = table_net(input_shape=(None,None,3),num_classes=2)
    model.compile(optimizer=Adam(lr=args.lr), loss=args.loss_name, metrics=args.metrics)

    model.fit_generator(
        trainloader,
        steps_per_epoch=max(1, len(trainP) // args.batchsize),
        callbacks=[checkpointer],
        validation_data=testloader,
        validation_steps=max(1, len(testP) // args.batchsize),
        epochs=30,
    )


def inference(args):
    # Init model
    detectron2 = init_detectron2(
        file_config=args.config_detectron2, file_checkpoint=args.model_detectron2
    )
    tabel_line = table_net(input_shape=(None, None, 3), num_classes=2)
    tabel_line.load_weights(args.model_table_line)

    # Load images
    image = cv2.imread(args.input_path)
    table_list, table_coords = make_prediction(image, detectron2)
    img = table_list[0]
    ceilboxes = detect_line(args=args, img=img, size=args.size, model=tabel_line)
    img_line, ls = draw_boxes_v2(img, ceilboxes)

    return ceilboxes, img_line


def main():
    parser = argparse.ArgumentParser(prog="Table Net")
    parser.add_argument(
        "-c", "--file_config", help="base config", default="./base_config.yaml"
    )
    subparsers = parser.add_subparsers(help="Actions", dest="action")
    train_parser = subparsers.add_parser("train", help="Start training process")
    infer_parser = subparsers.add_parser(
        "infer",
        help="Infer data with ONNX/TorchScript model, export report if label(s) provided",
    )

    args = vars(parser.parse_args())
    with open(args["file_config"], "r", encoding="utf8") as f:
        opt = yaml.safe_load(f)
    opt = AttrDict(opt)

    if args.get("action") == "train":
        train(opt)
    if args.get("action") == "infer":
        ceilboxes, img_line = inference(opt)
        print(ceilboxes)
        plt.imshow(img_line)
        plt.show()


if __name__ == "__main__":
    main()
