from tensorflow.keras import layers, models

def double_conv3d(inputs, n_filters, n_axis=4, n_kernel=3, padding="same"):
    conv = layers.Conv3D(n_filters, n_kernel, padding=padding)(inputs)
    conv = layers.BatchNormalization(axis=n_axis)(conv)
    conv = layers.LeakyReLU()(conv)
    conv = layers.Conv3D(n_filters, n_kernel, padding=padding)(conv)
    conv = layers.BatchNormalization(axis=n_axis)(conv)
    conv = layers.LeakyReLU()(conv)
    return conv

def first_double_conv3d(inputs, inputshape, n_filters, n_axis=4, n_kernel=3, padding="same"):
    conv = layers.Conv3D(n_filters, n_kernel, padding=padding, input_shape=inputshape)(inputs)
    conv = layers.BatchNormalization(axis=n_axis)(conv)
    conv = layers.LeakyReLU()(conv)
    conv = layers.Conv3D(n_filters, n_kernel, padding=padding)(conv)
    conv = layers.BatchNormalization(axis=n_axis)(conv)
    conv = layers.LeakyReLU()(conv)
    return conv

def pooling_3d(inputs, pool_size=(2,2,1)):
    pool = layers.MaxPooling3D(pool_size=pool_size)(inputs)
    return pool

def upsampling_3d(inputs, size=(2,2,1)):
    pool = layers.UpSampling3D(size=size)(inputs)
    return pool

def concat_3d(up, conv):
    return layers.Concatenate()([up, conv])

def output_conv3d(inputs):
    return layers.Conv3D(3,1, padding="same", activation="softmax")(inputs)

def first_step(inputs, inputshape, n_filters):
    layer = first_double_conv3d(inputs, inputshape, n_filters)
    pool = pooling_3d(layer)
    return layer,pool

def downsampling_step(inputs, n_filters):
    layer = double_conv3d(inputs, n_filters)
    pool = pooling_3d(layer)
    return layer, pool

def upsampling_step(inputs, conv_features, n_filters):
    x = upsampling_3d(inputs)
    x = concat_3d(x, conv_features)
    x = double_conv3d(x, n_filters)
    return x

def model_builder(shape=(128,128,10,1), n_filters=26):
    inputs = layers.Input(shape=shape)
    filters = [n_filters * (2 ** i) for i in range(5)]

    l1, p1 = first_step(inputs, shape, filters[0])
    l2, p2 = downsampling_step(p1, filters[1])
    l3, p3 = downsampling_step(p2, filters[2])
    l4, p4 = downsampling_step(p3, filters[3])
    
    l5 = double_conv3d(p4, filters[4])

    u1 = upsampling_step(l5, l4, filters[3])
    u2 = upsampling_step(u1, l3, filters[2])
    u3 = upsampling_step(u2, l2, filters[1])
    u4 = upsampling_step(u3, l1, filters[0])

    outputs = output_conv3d(u4)

    model = models.Model(inputs=inputs, outputs=outputs, name="U-Net")
    return model
