import tensorflow as tf
import keras
import cv2
import numpy as np
import os
import pandas as pd


def get_labels(filename):
    df = pd.read_csv(filename)
    for col_name, data in df.items():
        print("col_name:", col_name, "\ndata:", data)


def read_image(imageName, height=0, width=0):
    im = cv2.imread(imageName)
    if width != 0 and height != 0:
        im = cv2.resize(im, (height, width))
    return np.asarray(im, dtype=np.float32) / 255


def read_dir(imagePath, height, width, sort=False):
    dir = os.listdir(imagePath)
    if sort:
        dir = sorted(dir)
    dir_images = []
    for l in dir:
        dir_images.append(read_image(os.path.join(imagePath, l), height, width))
    return np.asarray(dir_images), dir


def get_model():
    base_model = tf.keras.applications.Xception(
        include_top=False,
        weights="imagenet",
        input_shape=(224, 224, 3),
        classes=1000,
    )
    inputs = keras.Input(shape=(224, 224, 3))
    x = base_model(inputs)
    x = keras.layers.GlobalAveragePooling2D()(x)
    outputs = keras.layers.Dense(50, activation="softmax")(x)
    model = keras.Model(inputs, outputs)
    model.compile(optimizer=keras.optimizers.Adam(1e-5),  # Very low learning rate
                  loss=keras.losses.CategoricalCrossentropy(),
                  metrics=[keras.metrics.Accuracy()])
    return model


train_x, names = read_dir("final_data/train", 224, 224)

val_x, vnames = read_dir("final_data/valid", 224, 224)

train_y = get_labels("final_data/train_labels.csv")
val_y = get_labels("final_data/valid_labels.csv")
# model.fit(new_dataset, epochs=20, callbacks=..., validation_data=...)


if __name__ == "__main__":
    pass
