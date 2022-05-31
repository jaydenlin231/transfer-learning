import os
import pathlib
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow import keras

NUM_CLASSES = 5

BATCH_SIZE = 32
IMG_HEIGHT = 256
IMG_WIDTH = 256

IMG_SIZE = (IMG_HEIGHT, IMG_WIDTH)
IMG_SHAPE = IMG_SIZE + (3,)

# -----------------------------------------------------------------------------

PROJECT_ROOT_DIR = "."
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "plots")
os.makedirs(IMAGES_PATH, exist_ok=True)

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)

# -----------------------------------------------------------------------------

def task_1():
    data_dir = pathlib.Path("./small_flower_dataset")
    return data_dir

# -----------------------------------------------------------------------------

def task_2():
    # Using the tf.keras.applications module download a pretrained MobileNetV2 network.
    base_model = keras.applications.MobileNetV2(
        input_shape = IMG_SHAPE,
        include_top = False, # Do not include the ImageNet classifier at the top.
        weights = "imagenet", # Load weights pre-trained on ImageNet.
    )

    # Freeze the base model.
    base_model.trainable = False
    return base_model


# -----------------------------------------------------------------------------

def task_3():
    inputs = keras.Input(shape=IMG_SHAPE)
    # We make sure that the base_model is running in inference mode here,
    # by passing `training=False`.
    x = base_model(inputs, training=False)
    # Convert features of shape `base_model.output_shape[1:]` to vectors
    x = keras.layers.GlobalAveragePooling2D()(x)

    # Replace the last layer with a Dense layer of the appropriate shape 
    # given that there are 5 classes in the small flower dataset.

    # A Dense classifier with 5 classes
    outputs = keras.layers.Dense(NUM_CLASSES, activation='softmax')(x)
    model = keras.Model(inputs, outputs)
    return model


# -----------------------------------------------------------------------------

# Task 4
def task_4():
    train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split = 0.3,
    subset = "training",
    seed = 123,
    image_size = IMG_SIZE,
    batch_size = BATCH_SIZE)

    val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split = 0.3,
    subset = "validation",
    seed = 123,
    image_size = IMG_SIZE,
    batch_size = BATCH_SIZE)

    test_ds = val_ds.take(4)
    val_ds = val_ds.skip(4)

    normalization_layer = tf.keras.layers.Rescaling(1./127.5, offset=-1)
    train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))
    test_ds = test_ds.map(lambda x, y: (normalization_layer(x), y))

    return train_ds, val_ds, test_ds


# -----------------------------------------------------------------------------
# Task 5 
def task_5():
    opt = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.0, nesterov=False)

    model.compile(
        optimizer= opt,
        loss=tf.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=['accuracy']
    )

    return model


def task_6():
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=15
    )

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    # plt.ylim([min(plt.ylim()),1])
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Cross Entropy')
    # plt.ylim([0,1.0])
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.show()

def task_7():
    old_weights = model.get_weights()
    learning_rates = [0.1, 0.01, 0.001]

    for rate in learning_rates:
        model.set_weights(old_weights)
        opt = tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.0, nesterov=True)

        model.compile(
            optimizer= opt,
            loss=tf.losses.SparseCategoricalCrossentropy(from_logits=False),
            metrics=['accuracy']
        )
        
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=15
        )

        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']

        loss = history.history['loss']
        val_loss = history.history['val_loss']

        plt.figure(figsize=(8, 8))
        plt.subplot(2, 1, 1)
        plt.plot(acc, label='Training Accuracy')
        plt.plot(val_acc, label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.ylabel('Accuracy')
        # plt.ylim([min(plt.ylim()),1])
        plt.title('Training and Validation Accuracy')

        plt.subplot(2, 1, 2)
        plt.plot(loss, label='Training Loss')
        plt.plot(val_loss, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.ylabel('Cross Entropy')
        # plt.ylim([0,1.0])
        plt.title('Training and Validation Loss')
        plt.xlabel('epoch')
        save_fig(f"{rate}_test")
        plt.show()




def task_8():
    old_weights = model.get_weights()
    momentum_tests = [0.25, 0.5, 0.75]
    lr = 0.001
    for momentum in momentum_tests:
        model.set_weights(old_weights)
        opt = tf.keras.optimizers.SGD(learning_rate = lr, momentum = momentum, nesterov=True)

        model.compile(
            optimizer= opt,
            loss=tf.losses.SparseCategoricalCrossentropy(from_logits=False),
            metrics=['accuracy']
        )
        
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=30
        )

        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']

        loss = history.history['loss']
        val_loss = history.history['val_loss']

        plt.figure(figsize=(8, 8))
        plt.subplot(2, 1, 1)
        plt.plot(acc, label='Training Accuracy')
        plt.plot(val_acc, label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.ylabel('Accuracy')
        # plt.ylim([min(plt.ylim()),1])
        plt.title('Training and Validation Accuracy')

        plt.subplot(2, 1, 2)
        plt.plot(loss, label='Training Loss')
        plt.plot(val_loss, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.ylabel('Cross Entropy')
        # plt.ylim([0,1.0])
        plt.title('Training and Validation Loss')
        plt.xlabel('epoch')
        save_fig(f"lr_{lr}_momentum_{momentum}_test2")
        # plt.show()


if __name__ == "__main__":
    data_dir = task_1()
    base_model = task_2()
    model = task_3()
    train_ds, val_ds, test_ds = task_4()
    model = task_5()
    # task_6()
    # task_7()
    task_8()

    pass



