import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from tensorflow import keras
from keras.layers import Dense



base_model = keras.applications.MobileNetV2(
    input_shape=None,
    alpha=1.0,
    include_top=False, # Do not include the ImageNet classifier at the top.
    weights="imagenet", # Load weights pre-trained on ImageNet.
    input_tensor=None,
    pooling=None,
    classes=1000,
    classifier_activation="softmax",
)

# freeze the base model.
base_model.trainable = False

print(base_model.summary())

inputs = keras.Input(shape=(150, 150, 3))
# We make sure that the base_model is running in inference mode here,
# by passing `training=False`. This is important for fine-tuning, as you will
# learn in a few paragraphs.
x = base_model(inputs, training=False)
# Convert features of shape `base_model.output_shape[1:]` to vectors
x = keras.layers.GlobalAveragePooling2D()(x)
# A Dense classifier with a single unit (binary classification)
outputs = keras.layers.Dense(1)(x)
model = keras.Model(inputs, outputs)


# builder = tfds.folder_dataset.ImageFolder('images/')
# print(builder.info)
# raw_train = builder.as_dataset(split='train', shuffle_files=True)
# raw_test = builder.as_dataset(split='test', shuffle_files=True)
# raw_valid = builder.as_dataset(split='valid', shuffle_files=True)
