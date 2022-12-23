"""
@author: nam157
"""
from typing import List

from tensorflow.keras.layers import (BatchNormalization, Conv2D, Input,
                                     LeakyReLU, MaxPooling2D, UpSampling2D,
                                     concatenate)
from tensorflow.keras.models import Model


def table_net(input_shape: List[int] = (512, 512, 3), num_classes: int = 1):
    inputs = Input(shape=input_shape)
    # 512
    use_bias = False
    down0a = Conv2D(16, (3, 3), padding="same", use_bias=use_bias)(inputs)
    down0a = BatchNormalization()(down0a)
    down0a = LeakyReLU(alpha=0.1)(down0a)
    down0a = Conv2D(16, (3, 3), padding="same", use_bias=use_bias)(down0a)
    down0a = BatchNormalization()(down0a)
    down0a = LeakyReLU(alpha=0.1)(down0a)
    down0a_pool = MaxPooling2D((2, 2), strides=(2, 2))(down0a)
    # 256

    down0 = Conv2D(32, (3, 3), padding="same", use_bias=use_bias)(down0a_pool)
    down0 = BatchNormalization()(down0)

    down0 = LeakyReLU(alpha=0.1)(down0)
    down0 = Conv2D(32, (3, 3), padding="same", use_bias=use_bias)(down0)
    down0 = BatchNormalization()(down0)
    down0 = LeakyReLU(alpha=0.1)(down0)
    down0_pool = MaxPooling2D((2, 2), strides=(2, 2))(down0)
    # 128

    down1 = Conv2D(64, (3, 3), padding="same", use_bias=use_bias)(down0_pool)
    down1 = BatchNormalization()(down1)
    down1 = LeakyReLU(alpha=0.1)(down1)
    down1 = Conv2D(64, (3, 3), padding="same", use_bias=use_bias)(down1)
    down1 = BatchNormalization()(down1)
    down1 = LeakyReLU(alpha=0.1)(down1)
    down1_pool = MaxPooling2D((2, 2), strides=(2, 2))(down1)
    # 64

    down2 = Conv2D(128, (3, 3), padding="same", use_bias=use_bias)(down1_pool)
    down2 = BatchNormalization()(down2)
    down2 = LeakyReLU(alpha=0.1)(down2)
    down2 = Conv2D(128, (3, 3), padding="same", use_bias=use_bias)(down2)
    down2 = BatchNormalization()(down2)
    down2 = LeakyReLU(alpha=0.1)(down2)
    down2_pool = MaxPooling2D((2, 2), strides=(2, 2))(down2)
    # 32

    down3 = Conv2D(256, (3, 3), padding="same", use_bias=use_bias)(down2_pool)
    down3 = BatchNormalization()(down3)
    down3 = LeakyReLU(alpha=0.1)(down3)
    down3 = Conv2D(256, (3, 3), padding="same", use_bias=use_bias)(down3)
    down3 = BatchNormalization()(down3)
    down3 = LeakyReLU(alpha=0.1)(down3)
    down3_pool = MaxPooling2D((2, 2), strides=(2, 2))(down3)
    # 16

    down4 = Conv2D(512, (3, 3), padding="same", use_bias=use_bias)(down3_pool)
    down4 = BatchNormalization()(down4)
    down4 = LeakyReLU(alpha=0.1)(down4)
    down4 = Conv2D(512, (3, 3), padding="same", use_bias=use_bias)(down4)
    down4 = BatchNormalization()(down4)
    down4 = LeakyReLU(alpha=0.1)(down4)
    down4_pool = MaxPooling2D((2, 2), strides=(2, 2))(down4)
    # 8

    center = Conv2D(1024, (3, 3), padding="same", use_bias=use_bias)(down4_pool)
    center = BatchNormalization()(center)
    center = LeakyReLU(alpha=0.1)(center)
    center = Conv2D(1024, (3, 3), padding="same", use_bias=use_bias)(center)
    center = BatchNormalization()(center)
    center = LeakyReLU(alpha=0.1)(center)
    # center

    up4 = UpSampling2D((2, 2))(center)
    up4 = concatenate([down4, up4], axis=3)
    up4 = Conv2D(512, (3, 3), padding="same", use_bias=use_bias)(up4)
    up4 = BatchNormalization()(up4)
    up4 = LeakyReLU(alpha=0.1)(up4)
    up4 = Conv2D(512, (3, 3), padding="same", use_bias=use_bias)(up4)
    up4 = BatchNormalization()(up4)
    up4 = LeakyReLU(alpha=0.1)(up4)
    up4 = Conv2D(512, (3, 3), padding="same", use_bias=use_bias)(up4)
    up4 = BatchNormalization()(up4)
    up4 = LeakyReLU(alpha=0.1)(up4)
    # 16

    up3 = UpSampling2D((2, 2))(up4)
    up3 = concatenate([down3, up3], axis=3)
    up3 = Conv2D(256, (3, 3), padding="same", use_bias=use_bias)(up3)
    up3 = BatchNormalization()(up3)
    up3 = LeakyReLU(alpha=0.1)(up3)
    up3 = Conv2D(256, (3, 3), padding="same", use_bias=use_bias)(up3)
    up3 = BatchNormalization()(up3)
    up3 = LeakyReLU(alpha=0.1)(up3)
    up3 = Conv2D(256, (3, 3), padding="same", use_bias=use_bias)(up3)
    up3 = BatchNormalization()(up3)
    up3 = LeakyReLU(alpha=0.1)(up3)
    # 32

    up2 = UpSampling2D((2, 2))(up3)
    up2 = concatenate([down2, up2], axis=3)
    up2 = Conv2D(128, (3, 3), padding="same", use_bias=use_bias)(up2)
    up2 = BatchNormalization()(up2)
    up2 = LeakyReLU(alpha=0.1)(up2)
    up2 = Conv2D(128, (3, 3), padding="same", use_bias=use_bias)(up2)
    up2 = BatchNormalization()(up2)
    up2 = LeakyReLU(alpha=0.1)(up2)
    up2 = Conv2D(128, (3, 3), padding="same", use_bias=use_bias)(up2)
    up2 = BatchNormalization()(up2)
    up2 = LeakyReLU(alpha=0.1)(up2)
    # 64

    up1 = UpSampling2D((2, 2))(up2)
    up1 = concatenate([down1, up1], axis=3)
    up1 = Conv2D(64, (3, 3), padding="same", use_bias=use_bias)(up1)
    up1 = BatchNormalization()(up1)
    up1 = LeakyReLU(alpha=0.1)(up1)
    up1 = Conv2D(64, (3, 3), padding="same", use_bias=use_bias)(up1)
    up1 = BatchNormalization()(up1)
    up1 = LeakyReLU(alpha=0.1)(up1)
    up1 = Conv2D(64, (3, 3), padding="same", use_bias=use_bias)(up1)
    up1 = BatchNormalization()(up1)
    up1 = LeakyReLU(alpha=0.1)(up1)
    # 128

    up0 = UpSampling2D((2, 2))(up1)
    up0 = concatenate([down0, up0], axis=3)
    up0 = Conv2D(32, (3, 3), padding="same", use_bias=use_bias)(up0)
    up0 = BatchNormalization()(up0)
    up0 = LeakyReLU(alpha=0.1)(up0)
    up0 = Conv2D(32, (3, 3), padding="same", use_bias=use_bias)(up0)
    up0 = BatchNormalization()(up0)
    up0 = LeakyReLU(alpha=0.1)(up0)
    up0 = Conv2D(32, (3, 3), padding="same", use_bias=use_bias)(up0)
    up0 = BatchNormalization()(up0)
    up0 = LeakyReLU(alpha=0.1)(up0)
    # 256

    up0a = UpSampling2D((2, 2))(up0)
    up0a = concatenate([down0a, up0a], axis=3)
    up0a = Conv2D(16, (3, 3), padding="same", use_bias=use_bias)(up0a)
    up0a = BatchNormalization()(up0a)
    up0a = LeakyReLU(alpha=0.1)(up0a)
    up0a = Conv2D(16, (3, 3), padding="same", use_bias=use_bias)(up0a)
    up0a = BatchNormalization()(up0a)
    up0a = LeakyReLU(alpha=0.1)(up0a)
    up0a = Conv2D(16, (3, 3), padding="same", use_bias=use_bias)(up0a)
    up0a = BatchNormalization()(up0a)
    up0a = LeakyReLU(alpha=0.1)(up0a)
    # 512

    classify = Conv2D(num_classes, (1, 1), activation="sigmoid")(up0a)

    model = Model(inputs=inputs, outputs=classify)

    return model


if __name__ == "__main__":
    model = table_net()
    print(model.summary())
