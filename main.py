import tensorflow as tf
import keras
import cv2
import numpy as np
import os
import pandas as pd
from keras.utils.np_utils import to_categorical


def get_labels(filename):
    df = pd.read_csv(filename)
    return to_categorical(df['label'].tolist())


def save_predictions(lst):
    df = pd.DataFrame(lst, columns=['results'])
    df.to_csv("res.csv", index=False)


def read_image(imageName, height=0, width=0):
    im = cv2.imread(imageName)
    if width != 0 and height != 0:
        im = cv2.resize(im, (height, width))
    return np.asarray(im, dtype=np.float32) / 255


def read_dir(imagePath, height, width, sort=False):
    dir = os.listdir(imagePath)
    if sort:
        dir = sorted(dir, key=lambda x: int(x[:-4]))
    dir_images = []
    for l in dir:
        dir_images.append(read_image(os.path.join(imagePath, l), height, width))
    return np.asarray(dir_images), dir


def get_model():
    base_model = tf.keras.applications.Xception(
        include_top=False,
        weights="imagenet",
        input_shape=(224, 224, 3),
        #classes=1000,
    )
    inputs = keras.Input(shape=(224, 224, 3))
    x = base_model(inputs)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    outputs = tf.keras.layers.Dense(50, activation="softmax")(x)
    model = keras.Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),  # Very low learning rate
                  loss=tf.keras.losses.CategoricalCrossentropy(),
                  metrics="accuracy")
    return model


train_x, names = read_dir("final_data/train", 224, 224, True)

val_x, vnames = read_dir("final_data/valid", 224, 224, True)

train_y = get_labels("final_data/train_labels.csv")
val_y = get_labels("final_data/valid_labels.csv")

model = get_model()
#model.fit(x=train_x, y=train_y, epochs=20, batch_size=16, validation_data=(val_x, val_y))
#model.save_weights("weights.h5")
model.load_weights("weights.h5")

test_x, _ = read_dir("final_data/test", 224, 224, True)

results = model.predict(test_x)
results = np.argmax(results, axis=1)
save_predictions(results)
if __name__ == "__main__":
    pass
