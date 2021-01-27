"""
Note:
    1.This file is used for designing the structure of encoder and decoder.
    2.The neural network structure in this model file is CsiNet, more details about CsiNet can be found in [1].
[1] C. Wen, W. Shih and S. Jin, "Deep Learning for Massive MIMO CSI Feedback", in IEEE Wireless Communications Letters, vol. 7, no. 5, pp. 748-751, Oct. 2018, doi: 10.1109/LWC.2018.2818160.
    3.The output of the encoder must be the bitstream.
"""

#=======================================================================================================================
#=======================================================================================================================
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
#=======================================================================================================================
#=======================================================================================================================
# Number to Bit Defining Function Defining
def Num2Bit(Num, B):
    Num_ = Num.numpy()
    bit = (np.unpackbits(np.array(Num_, np.uint8), axis=1).reshape(-1, Num_.shape[1], 8)[:, :, (8-B):]).reshape(-1,
                                                                                                            Num_.shape[
                                                                                                                1] * B)
    bit.astype(np.float32)
    return tf.convert_to_tensor(bit, dtype=tf.float32)
# Bit to Number Function Defining
def Bit2Num(Bit, B):
    Bit_ = Bit.numpy()
    Bit_.astype(np.float32)
    Bit_ = np.reshape(Bit_, [-1, int(Bit_.shape[1] / B), B])
    num = np.zeros(shape=np.shape(Bit_[:, :, 1]))
    for i in range(B):
        num = num + Bit_[:, :, i] * 2 ** (B - 1 - i)
    return tf.cast(num, dtype=tf.float32)
#=======================================================================================================================
#=======================================================================================================================
# Quantization and Dequantization Layers Defining
@tf.custom_gradient
def QuantizationOp(x, B):
    step = tf.cast((2 ** B), dtype=tf.float32)
    result = tf.cast((tf.round(x * step - 0.5)), dtype=tf.float32)
    result = tf.py_function(func=Num2Bit, inp=[result, B], Tout=tf.float32)
    def custom_grad(dy):
        grad = dy
        return (grad, grad)
    return result, custom_grad
class QuantizationLayer(tf.keras.layers.Layer):
    def __init__(self, B,**kwargs):
        self.B = B
        super(QuantizationLayer, self).__init__()
    def call(self, x):
        return QuantizationOp(x, self.B)
    def get_config(self):
        # Implement get_config to enable serialization. This is optional.
        base_config = super(QuantizationLayer, self).get_config()
        base_config['B'] = self.B
        return base_config
@tf.custom_gradient
def DequantizationOp(x, B):
    x = tf.py_function(func=Bit2Num, inp=[x, B], Tout=tf.float32)
    step = tf.cast((2 ** B), dtype=tf.float32)
    result = tf.cast((x + 0.5) / step, dtype=tf.float32)
    def custom_grad(dy):
        grad = dy * 1
        return (grad, grad)
    return result, custom_grad
class DeuantizationLayer(tf.keras.layers.Layer):
    def __init__(self, B,**kwargs):
        self.B = B
        super(DeuantizationLayer, self).__init__()
    def call(self, x):
        return DequantizationOp(x, self.B)
    def get_config(self):
        base_config = super(DeuantizationLayer, self).get_config()
        base_config['B'] = self.B
        return base_config
#=======================================================================================================================
#=======================================================================================================================
img_height = 24
img_width = 16
img_channels = 2
img_total = img_height*img_width*img_channels
# network params
dense_num = 2
encoded_dim = 480  # compress rate=1/4->dim.=512, compress rate=1/16->dim.=128, compress rate=1/32->dim.=64, compress rate=1/64->dim.=32
compression_rate = 1 # positive integer, 1 causes no reduction
non_local_mode = 'embedded' # 'gaussian' or 'embedded'
from non_local import non_local_block

# Encoder and Decoder Function Defining
def Encoder(enc_input, num_feedback_bits):
    num_quan_bits = 4
    x = tf.transpose(enc_input, perm=[0,3,1,2])
    x = tf.keras.layers.Conv2D(2, (3, 3), padding='same', data_format="channels_first")(x)
    x = add_common_layers(x)
    x = non_local_block(x, compression=compression_rate, mode=non_local_mode)
    x = tf.keras.layers.Reshape((img_total,))(x)
    x = layers.Dense(units=int(num_feedback_bits / num_quan_bits), activation='sigmoid')(x)
    enc_output = QuantizationLayer(num_quan_bits)(x)
    return enc_output

def add_common_layers(y):
    y = tf.keras.layers.BatchNormalization()(y)
    y = tf.keras.layers.LeakyReLU()(y)
    return y
def dense_residual_block(y):
    layers_concat = list()
    layers_concat.append(y)
    y = add_common_layers(y)
    y = tf.keras.layers.Conv2D(8, (3, 3), padding='same', data_format="channels_first")(y)
    layers_concat.append(y)
    y = tf.concat(layers_concat, axis=1)
    y = add_common_layers(y)
    y = tf.keras.layers.Conv2D(16, (3, 3), padding='same', data_format="channels_first")(y)
    layers_concat.append(y)
    y = tf.concat(layers_concat, axis = 1)
    y = add_common_layers(y)
    y = tf.keras.layers.Conv2D(2, (3, 3), padding='same', data_format="channels_first")(y)
    layers_concat.append(y)
    y = tf.concat(layers_concat, axis=1)
    return y

def Decoder(dec_input,num_feedback_bits):
    num_quan_bits = 4
    x = DeuantizationLayer(num_quan_bits)(dec_input)
    x = tf.keras.layers.Reshape((int(num_feedback_bits/num_quan_bits),))(x)
    x = layers.Dense(img_total, activation='sigmoid')(x)
    x = layers.Reshape((2, 24, 16))(x)
    x = non_local_block(x, compression=compression_rate, mode=non_local_mode)
    for i in range(dense_num):
        x = dense_residual_block(x)
        x = add_common_layers(x)
        x = tf.keras.layers.Conv2D(2, (1, 1), padding='same', data_format="channels_first")(x)
    x = tf.keras.layers.Conv2D(2, (3, 3), activation='sigmoid', padding='same', data_format="channels_first")(x)
    dec_output = tf.transpose(x, perm=[0,2,3,1])
    return dec_output
#=======================================================================================================================
#=======================================================================================================================
# NMSE Function Defining
def NMSE(x, x_hat):
    x_real = np.reshape(x[:, :, :, 0], (len(x), -1))
    x_imag = np.reshape(x[:, :, :, 1], (len(x), -1))
    x_hat_real = np.reshape(x_hat[:, :, :, 0], (len(x_hat), -1))
    x_hat_imag = np.reshape(x_hat[:, :, :, 1], (len(x_hat), -1))
    x_C = x_real - 0.5 + 1j * (x_imag - 0.5)
    x_hat_C = x_hat_real - 0.5 + 1j * (x_hat_imag - 0.5)
    power = np.sum(abs(x_C) ** 2, axis=1)
    mse = np.sum(abs(x_C - x_hat_C) ** 2, axis=1)
    nmse = np.mean(mse / power)
    return nmse
def get_custom_objects():
    return {"QuantizationLayer":QuantizationLayer,"DeuantizationLayer":DeuantizationLayer}
#     return {"QuantizationLayer":QuantizationLayer,"DeuantizationLayer":DeuantizationLayer,"ConvBN":ConvBN,"Mish":Mish,"ResBlock":ResBlock,"Encoder_conv":Encoder_conv,"CRBlock64":CRBlock64,"CRBlock":CRBlock}
#=======================================================================================================================
#=======================================================================================================================
#自定义metrics
class NMSE_tf(keras.metrics.Metric):
    def __init__(self, name="nmse", **kwargs):
        super(NMSE_tf, self).__init__(name=name, **kwargs)
        self.totalLoss = self.add_weight(name="totalLoss", initializer="zeros")
        self.totalCount = self.add_weight(name="totalCount", dtype=tf.int32, initializer="zeros")
    def update_state(self, y_true, y_pred):
        x_real = tf.reshape(y_true[:, :, :, 0], (tf.shape(y_true)[0], -1))-0.5
        x_imag = tf.reshape(y_true[:, :, :, 1], (tf.shape(y_true)[0], -1))-0.5
        x_hat_real = tf.reshape(y_pred[:, :, :, 0], (tf.shape(y_pred)[0], -1))-0.5
        x_hat_imag = tf.reshape(y_pred[:, :, :, 1], (tf.shape(y_pred)[0], -1))-0.5
        power = tf.reduce_sum(x_real ** 2 + x_imag ** 2, axis=1)
        mse = tf.reduce_sum((x_real - x_hat_real) ** 2 + (x_imag - x_hat_imag) ** 2, axis=1)
        nmse = tf.reduce_sum(mse / power)
        self.totalCount.assign_add(tf.shape(y_true)[0])
        self.totalLoss.assign_add(nmse)
    def result(self):
        return self.totalLoss / tf.cast(self.totalCount, tf.float32)#必须要转化成一样的类型才能相除
    def reset_states(self):
        # The state of the metric will be reset at the start of each epoch.
        self.totalLoss.assign(0.0)
        self.totalCount.assign(0)
#自定义callback用于评估与保存
class CheckPointer(keras.callbacks.Callback):
    """自定义评估与保存
    """
    def __init__(self, valid_generator):
        self.valid_generator = valid_generator
        self.best_val_nmse = 0.2
    def evaluate(self, data):
        y_true = data #autoencoder中y_true就是输入数据
        y_pred = autoencoderModel.predict(y_true, batch_size=512)
        res = NMSE(y_true, y_pred)
        return res
    def on_epoch_end(self, epoch, logs=None):
        val_nmse = self.evaluate(self.valid_generator)
        if val_nmse < self.best_val_nmse:
            self.best_val_nmse = val_nmse
            # Encoder Saving
            encModel.save('./modelSubmit_2/encoder.h5')
            # Decoder Saving
            decModel.save('./modelSubmit_2/decoder.h5')
            print("tf model saved!")
        print('\nval NMSE = ' + np.str(val_nmse))
def non_local_block(ip, intermediate_dim=None, compression=2,
                    mode='embedded', add_residual=True):
    """
    Adds a Non-Local block for self attention to the input tensor.
    Input tensor can be or rank 3 (temporal), 4 (spatial) or 5 (spatio-temporal).
    Arguments:
        ip: input tensor
        intermediate_dim: The dimension of the intermediate representation. Can be
            `None` or a positive integer greater than 0. If `None`, computes the
            intermediate dimension as half of the input channel dimension.
        compression: None or positive integer. Compresses the intermediate
            representation during the dot products to reduce memory consumption.
            Default is set to 2, which states halve the time/space/spatio-time
            dimension for the intermediate step. Set to 1 to prevent computation
            compression. None or 1 causes no reduction.
        mode: Mode of operation. Can be one of `embedded`, `gaussian`, `dot` or
            `concatenate`.
        add_residual: Boolean value to decide if the residual connection should be
            added or not. Default is True for ResNets, and False for Self Attention.
    Returns:
        a tensor of same shape as input
    """
    channel_dim = 1 if keras.backend.image_data_format() == 'channels_first' else -1
    ip_shape = keras.backend.int_shape(ip)

    if mode not in ['gaussian', 'embedded', 'dot', 'concatenate']:
        raise ValueError('`mode` must be one of `gaussian`, `embedded`, `dot` or `concatenate`')

    if compression is None:
        compression = 1

    dim1, dim2, dim3 = None, None, None

    # check rank and calculate the input shape
    if len(ip_shape) == 3:  # temporal / time series data
        rank = 3
        batchsize, dim1, channels = ip_shape

    elif len(ip_shape) == 4:  # spatial / image data
        rank = 4

        if channel_dim == 1:
            batchsize, channels, dim1, dim2 = ip_shape
        else:
            batchsize, dim1, dim2, channels = ip_shape

    elif len(ip_shape) == 5:  # spatio-temporal / Video or Voxel data
        rank = 5

        if channel_dim == 1:
            batchsize, channels, dim1, dim2, dim3 = ip_shape
        else:
            batchsize, dim1, dim2, dim3, channels = ip_shape

    else:
        raise ValueError('Input dimension has to be either 3 (temporal), 4 (spatial) or 5 (spatio-temporal)')

    # verify correct intermediate dimension specified
    if intermediate_dim is None:
        intermediate_dim = channels // 2

        if intermediate_dim < 1:
            intermediate_dim = 1

    else:
        intermediate_dim = int(intermediate_dim)

        if intermediate_dim < 1:
            raise ValueError('`intermediate_dim` must be either `None` or positive integer greater than 1.')

    if mode == 'gaussian':  # Gaussian instantiation
        x1 = tf.keras.layers.Reshape((-1, channels))(ip)  # xi
        x2 = tf.keras.layers.Reshape((-1, channels))(ip)  # xj
        f = tf.keras.layers.dot([x1, x2], axes=2)
        f = tf.keras.layers.Activation('softmax')(f)

    elif mode == 'dot':  # Dot instantiation
        # theta path
        theta = _convND(ip, rank, intermediate_dim)
        theta = tf.keras.layers.Reshape((-1, intermediate_dim))(theta)

        # phi path
        phi = _convND(ip, rank, intermediate_dim)
        phi = tf.keras.layers.Reshape((-1, intermediate_dim))(phi)

        f = tf.keras.layers.dot([theta, phi], axes=2)

        size = K.int_shape(f)

        # scale the values to make it size invariant
        f = tf.keras.layers.Lambda(lambda z: (1. / float(size[-1])) * z)(f)

    elif mode == 'concatenate':  # Concatenation instantiation
        raise NotImplementedError('Concatenate model has not been implemented yet')

    else:  # Embedded Gaussian instantiation
        # theta path
        theta = _convND(ip, rank, intermediate_dim)
        theta = tf.keras.layers.Reshape((-1, intermediate_dim))(theta)

        # phi path
        phi = _convND(ip, rank, intermediate_dim)
        phi = tf.keras.layers.Reshape((-1, intermediate_dim))(phi)

        if compression > 1:
            # shielded computation
            phi = tf.keras.layers.MaxPool1D(compression)(phi)

        f = tf.keras.layers.dot([theta, phi], axes=2)
        f = tf.keras.layers.Activation('softmax')(f)

    # g path
    g = _convND(ip, rank, intermediate_dim)
    g = tf.keras.layers.Reshape((-1, intermediate_dim))(g)

    if compression > 1 and mode == 'embedded':
        # shielded computation
        g = tf.keras.layers.MaxPool1D(compression)(g)

    # compute output path
    y = tf.keras.layers.dot([f, g], axes=[2, 1])

    # reshape to input tensor format
    if rank == 3:
        y = tf.keras.layers.Reshape((dim1, intermediate_dim))(y)
    elif rank == 4:
        if channel_dim == -1:
            y = tf.keras.layers.Reshape((dim1, dim2, intermediate_dim))(y)
        else:
            y = tf.keras.layers.Reshape((intermediate_dim, dim1, dim2))(y)
    else:
        if channel_dim == -1:
            y = tf.keras.layers.Reshape((dim1, dim2, dim3, intermediate_dim))(y)
        else:
            y = tf.keras.layers.Reshape((intermediate_dim, dim1, dim2, dim3))(y)

    # project filters
    y = _convND(y, rank, channels)

    # residual connection
    if add_residual:
        y = tf.keras.layers.add([ip, y])

    return y

def _convND(ip, rank, channels):
    assert rank in [3, 4, 5], "Rank of input must be 3, 4 or 5"
    if rank == 3:
        x = tf.keras.layers.Conv1D(channels, 1, padding='same', use_bias=False, kernel_initializer='he_normal')(ip)
    elif rank == 4:
        x = tf.keras.layers.Conv2D(channels, (1, 1), padding='same', use_bias=False, kernel_initializer='he_normal')(ip)
    else:
        x = tf.keras.layers.Conv3D(channels, (1, 1, 1), padding='same', use_bias=False, kernel_initializer='he_normal')(ip)
    return x