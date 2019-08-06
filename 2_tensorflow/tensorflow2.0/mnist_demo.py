# coding: utf-8
# File: mnist_demo.py
# Author: zhoubin
# Date: 20190806

import os
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

# 指定用哪块GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data() # 把mnist.npz放到~/.keras/datasets下
x_train = x_train.reshape(60000, 784).astype('float32') /255
x_test = x_test.reshape(10000, 784).astype('float32') /255

def get_compiled_model():
    inputs = keras.Input(shape=(784,), name='mnist_input')
    h1 = layers.Dense(64, activation='relu')(inputs)
    h2 = layers.Dense(64, activation='relu')(h1)
    outputs = layers.Dense(10, activation='softmax')(h2)
    model = keras.Model(inputs, outputs)
    model.compile(optimizer=keras.optimizers.RMSprop(),
                 loss=keras.losses.SparseCategoricalCrossentropy(),
                 metrics=[keras.metrics.SparseCategoricalAccuracy()])
    return model

model = get_compiled_model()
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(64)

val_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
val_dataset = val_dataset.batch(64)

# model.fit(train_dataset, epochs=3)
# steps_per_epoch 每个epoch只训练几步
# validation_steps 每次验证，验证几步
model.fit(train_dataset, epochs=5, steps_per_epoch=100, validation_data=val_dataset, validation_steps=3)