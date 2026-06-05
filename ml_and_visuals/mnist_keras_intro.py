"""
mnist_keras_intro.py — MNIST + Keras "first neural net" warm-up (incomplete).

A scratch script from working through a deep-learning intro: loads MNIST, builds
a tiny dense classifier (512 relu -> 10 softmax), compiles it, and flattens /
normalizes the images. NOTE: it stops before model.fit(), so as written it only
sets things up and prints shapes — it does not actually train. Kept as-is for
posterity; add e.g. `model.fit(train_images, train_labels, epochs=5,
batch_size=128)` to finish it.
"""

import os
os.environ["KERAS_BACKEND"] = "tensorflow"


from keras.datasets import mnist

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

print(train_images.shape)
print(len(train_labels))
print(train_labels)
print(test_images.shape)
print(len(test_labels))
print(test_labels)

import keras
from keras import layers

model = keras.Sequential(
    [
        layers.Dense(512, activation="relu"),
        layers.Dense(10, activation="softmax"),
    ]
)

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)


train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype("float32") / 255
print(train_images.shape)
test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype("float32") / 255
print(test_images.shape)

print(train_images.size)