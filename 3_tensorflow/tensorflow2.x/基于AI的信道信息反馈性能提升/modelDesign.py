"""
Note:
    1.This file is used for designing the structure of encoder and decoder.
    2.The neural network structure in this model file is CsiNet, more details about CsiNet can be found in [1].
[1] C. Wen, W. Shih and S. Jin, "Deep Learning for Massive MIMO CSI Feedback", in IEEE Wireless Communications Letters, vol. 7, no. 5, pp. 748-751, Oct. 2018, doi: 10.1109/LWC.2018.2818160.
    3.The output of the encoder must be the bitstream.
"""

# =======================================================================================================================
# =======================================================================================================================
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
# =======================================================================================================================
# =======================================================================================================================
# Number to Bit Defining Function Defining
def Num2Bit(Num, B):
    Num_ = Num.numpy()
    bit = (np.unpackbits(np.array(Num_, np.uint8), axis=1).reshape(-1, Num_.shape[1], 8)[:, :, (8 - B):]).reshape(-1,
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


# =======================================================================================================================
# =======================================================================================================================
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
    def __init__(self, B, **kwargs):
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
    def __init__(self, B, **kwargs):
        self.B = B
        super(DeuantizationLayer, self).__init__()

    def call(self, x):
        return DequantizationOp(x, self.B)

    def get_config(self):
        base_config = super(DeuantizationLayer, self).get_config()
        base_config['B'] = self.B
        return base_config


# =======================================================================================================================
# =======================================================================================================================
# Encoder and Decoder Function Defining
channelNum = 109


class Mish(layers.Layer):
    def __init__(self):
        super(Mish, self).__init__()

    def call(self, x):
        x = x * (tf.tanh(tf.nn.softplus(x)))
        return x

    def get_config(self):  # 在有自定义网络层时，需要保存模型时，重写get_config函数(与init时传入的参数一致)
        return {}


def conv3x3(out_planes, strides=1):
    """3x3 convolution with padding"""
    return layers.Conv2D(out_planes, kernel_size=(3, 3), strides=1, padding='SAME', data_format='channels_last')


class ConvBN(layers.Layer):
    def __init__(self, out_planes, kernel_size, stride=1, data_format='channels_last'):
        super(ConvBN, self).__init__()
        self.out_planes = out_planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.data_format = data_format

    def build(self, input_shape):
        self.conv = layers.Conv2D(self.out_planes, kernel_size=self.kernel_size, strides=self.stride,
                                  data_format=self.data_format, padding='SAME')
        self.bn = layers.BatchNormalization(axis=3)  # axis=3对应`data_format="channels_first"`
        self.Mish = Mish()

    def call(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.Mish(x)
        return x

    def get_config(self):
        config = {"out_planes": self.out_planes, "kernel_size": self.kernel_size, "stride": self.stride,
                  "data_format": self.data_format}  # (与init时传入的参数一致)
        #         #一般是用注释的这种方式config中需要包含name，可能是tf版本问题不需要name,trainable等
        #         base_config = super(ConvBN, self).get_config()
        #         return dict(list(base_config.items()) + list(config.items()))
        return config


class ResBlock(layers.Layer):
    """
    Sequential residual blocks each of which consists of \
    two convolution layers.
    Args:
        ch (int): number of input and output channels.
        nblocks (int): number of residual blocks.
        shortcut (bool): if True, residual tensor addition is enabled.
    """

    def __init__(self, ch, nblocks=1, shortcut=True):
        super().__init__()
        self.shortcut = shortcut
        self.nblocks = nblocks
        self.ch = ch

    def build(self, input_shape):
        self.module_list = []
        for i in range(self.nblocks):
            resblock_one = []
            resblock_one.append(ConvBN(self.ch, 1))
            resblock_one.append(Mish())
            resblock_one.append(ConvBN(self.ch, 3))
            resblock_one.append(Mish())
            self.module_list.append(list(resblock_one))

    def call(self, x):
        for module in self.module_list:
            h = x
            for res in module:
                h = res(h)
            x = x + h if self.shortcut else h
        return x

    def get_config(self):
        config = {"shortcut": self.shortcut, "nblocks": self.nblocks, "ch": self.ch}
        return config


class Encoder_conv(layers.Layer):
    def __init__(self, in_planes=128, blocks=2):
        super().__init__()
        self.in_planes = in_planes
        self.blocks = blocks

    def build(self, input_shape):
        self.conv2 = ConvBN(self.in_planes, [1, 9])
        self.conv3 = ConvBN(self.in_planes, [9, 1])
        self.conv4 = ConvBN(self.in_planes, 1)
        self.resBlock = ResBlock(ch=self.in_planes, nblocks=self.blocks)
        self.conv5 = ConvBN(self.in_planes, [1, 7])
        self.conv6 = ConvBN(self.in_planes, [7, 1])
        self.conv7 = ConvBN(self.in_planes, 1)
        self.relu = Mish()

    def call(self, input):
        x2 = self.conv2(input)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        r1 = self.resBlock(x4)
        x5 = self.conv5(r1)
        x6 = self.conv6(x5)
        x7 = self.conv7(x6)
        x7 = self.relu(x7 + x4)
        return x7

    def get_config(self):
        config = {"in_planes": self.in_planes, "blocks": self.blocks}
        return config


class CRBlock64(layers.Layer):
    def __init__(self):
        super(CRBlock64, self).__init__()

    def build(self, input_shape):
        self.convbncrb = ConvBN(channelNum * 2, 3)
        self.path1 = Encoder_conv(channelNum * 2, 4)
        self.path2 = keras.Sequential([
            ConvBN(channelNum * 2, [1, 5]),
            ConvBN(channelNum * 2, [5, 1]),
            ConvBN(channelNum * 2, 1),
            ConvBN(channelNum * 2, 3),
        ])
        self.encoder_conv = Encoder_conv(channelNum * 4, 4)
        self.encoder_conv1 = ConvBN(channelNum, 1)
        self.relu = Mish()

    def call(self, x):
        identity = tf.identity(x)
        x = self.convbncrb(x)
        out1 = self.path1(x)
        out2 = self.path2(x)
        out = tf.concat([out1, out2], axis=3)
        out = self.relu(out)
        out = self.encoder_conv(out)
        out = self.encoder_conv1(out)
        out = self.relu(out + identity)
        return out

    def get_config(self):
        return {}


class CRBlock(layers.Layer):
    def __init__(self):
        super(CRBlock, self).__init__()

    def build(self, input_shape):
        self.convban = ConvBN(channelNum, 3)
        self.path1 = Encoder_conv(channelNum, 4)
        self.path2 = keras.Sequential([
            ConvBN(channelNum, [1, 5]),
            ConvBN(channelNum, [5, 1]),
            ConvBN(channelNum, 1),
        ])
        self.encoder_conv = Encoder_conv(channelNum * 2)
        self.encoder_conv1 = ConvBN(channelNum, 1)
        self.relu = Mish()

    def call(self, x):
        identity = tf.identity(x)
        x = self.convban(x)
        out1 = self.path1(x)
        out2 = self.path2(x)
        out = tf.concat([out1, out2], axis=3)
        out = self.relu(out)
        out = self.encoder_conv(out)
        out = self.encoder_conv1(out)
        out = self.relu(out + identity)
        return out

    def get_config(self):
        return {}


def Encoder(enc_input, num_feedback_bits):
    num_quan_bits = 4
    x = ConvBN(channelNum, 3)(enc_input)
    h1 = Encoder_conv(channelNum)(x)
    h2 = keras.Sequential([ConvBN(channelNum, [1, 5]), ConvBN(channelNum, [5, 1]), ConvBN(channelNum, 3)])(x)
    h = tf.concat([h1, h2], axis=3)
    h = Encoder_conv(channelNum * 2)(h)
    h = ConvBN(2, 1)(h)
    h = layers.Flatten()(h)
    h = layers.Dense(units=int(num_feedback_bits / num_quan_bits), activation='sigmoid')(h)
    enc_output = QuantizationLayer(num_quan_bits)(h)
    return enc_output


def Decoder(dec_input, num_feedback_bits):
    num_quan_bits = 4
    h = DeuantizationLayer(num_quan_bits)(dec_input)
    h = tf.keras.layers.Reshape((-1, int(num_feedback_bits / num_quan_bits)))(h)
    h = layers.Dense(768, activation='sigmoid')(h)
    h = layers.Reshape((24, 16, 2))(h)
    h = keras.Sequential([ConvBN(channelNum, 3), CRBlock64(), CRBlock()])(h)
    h = conv3x3(2)(h)
    dec_output = tf.sigmoid(h)
    return dec_output


# =======================================================================================================================
# =======================================================================================================================
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


# 自定义metrics
class NMSE_tf(keras.metrics.Metric):
    def __init__(self, name="nmse", **kwargs):
        super(NMSE_tf, self).__init__(name=name, **kwargs)
        self.totalLoss = self.add_weight(name="totalLoss", initializer="zeros")
        self.totalCount = self.add_weight(name="totalCount", dtype=tf.int32, initializer="zeros")

    def update_state(self, y_true, y_pred):
        x_real = tf.reshape(y_true[:, :, :, 0], (tf.shape(y_true)[0], -1)) - 0.5
        x_imag = tf.reshape(y_true[:, :, :, 1], (tf.shape(y_true)[0], -1)) - 0.5
        x_hat_real = tf.reshape(y_pred[:, :, :, 0], (tf.shape(y_pred)[0], -1)) - 0.5
        x_hat_imag = tf.reshape(y_pred[:, :, :, 1], (tf.shape(y_pred)[0], -1)) - 0.5
        power = tf.reduce_sum(x_real ** 2 + x_imag ** 2, axis=1)
        mse = tf.reduce_sum((x_real - x_hat_real) ** 2 + (x_imag - x_hat_imag) ** 2, axis=1)
        nmse = tf.reduce_sum(mse / power)
        self.totalCount.assign_add(tf.shape(y_true)[0])
        self.totalLoss.assign_add(nmse)

    def result(self):
        return self.totalLoss / tf.cast(self.totalCount, tf.float32)  # 必须要转化成一样的类型才能相除

    def reset_states(self):
        # The state of the metric will be reset at the start of each epoch.
        self.totalLoss.assign(0.0)
        self.totalCount.assign(0)


# 自定义callback用于评估与保存
class CheckPointer(keras.callbacks.Callback):
    """自定义评估与保存
    """

    def __init__(self, valid_generator):
        self.valid_generator = valid_generator
        self.best_val_nmse = 1

    def evaluate(self, data):
        y_true = data  # autoencoder中y_true就是输入数据
        y_pred = autoencoderModel.predict(y_true, batch_size=512)
        res = NMSE(y_true, y_pred)
        return res

    def on_epoch_end(self, epoch, logs=None):
        val_nmse = self.evaluate(self.valid_generator)
        if val_nmse < self.best_val_nmse:
            self.best_val_nmse = val_nmse
            # Encoder Saving
            encModel.save('./modelSubmit/encoder.h5')
            # Decoder Saving
            decModel.save('./modelSubmit/decoder.h5')
            print("tf model saved!")
        print('\nval NMSE = ' + np.str(val_nmse))


def get_custom_objects():
    return {"QuantizationLayer": QuantizationLayer, "DeuantizationLayer": DeuantizationLayer, "ConvBN": ConvBN,
            "Mish": Mish, "ResBlock": ResBlock, "Encoder_conv": Encoder_conv, "CRBlock64": CRBlock64,
            "CRBlock": CRBlock}
# =======================================================================================================================
# =======================================================================================================================
