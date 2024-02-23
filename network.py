import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.keras import layers
import numpy as np
import matplotlib.pyplot as plt

loss_fn = tf.keras.losses.BinaryCrossentropy()
optimiser = tf.keras.optimizers.Adam()

#Try Batch normalisation + leaky ReLU
def double_conv3d(inputs, input_shape=None, n_filters=64, n_kernel=3, dropout=None, activation="relu", padding="same"):
    conv = layers.Conv3D(n_filters, n_kernel, activation=activation, padding=padding, input_shape=input_shape)(inputs)
    conv = layers.Conv3D(n_filters, n_kernel, activation=activation, padding=padding, input_shape=input_shape)(conv)
    return conv

def pooling_3d(inputs, pool_size=(2,2,2)): #Max pooling
    pool = layers.MaxPooling3D(pool_size=pool_size)(inputs)
    return pool

def upsampling_3d(inputs, size=(2,2,2)): #Max pooling
    pool = layers.UpSampling3D(size=size)(inputs)
    return pool

def concat_3d(conv, up):
    return layers.concatenate([conv, up], axis=3)

def output_conv3d(inputs):
    return layers.Conv3D(1,1,activation="sigmoid")(inputs)