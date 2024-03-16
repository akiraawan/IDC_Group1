import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, BatchNormalization, LeakyReLU, GaussianNoise

def hidden_layer(inputs, units=32, n_axis=1, stddev=0.1):
    layer = Dense(units)(inputs)
    layer = BatchNormalization(axis=n_axis)(layer)
    layer = LeakyReLU()(layer)
    layer = GaussianNoise(stddev)
    return layer

def output_layer(inputs, units=5):
    return Dense(units, activation='softmax')(inputs)

def model_builder(input_shape):
    inputs = Input(shape=input_shape)
    l1 = hidden_layer(inputs)
    l2 = hidden_layer(l1)
    l3 = hidden_layer(l2)
    l4 = hidden_layer(l3)
    outputs = output_layer(l4)

    model = Model(inputs=inputs, outputs=outputs)
    return model