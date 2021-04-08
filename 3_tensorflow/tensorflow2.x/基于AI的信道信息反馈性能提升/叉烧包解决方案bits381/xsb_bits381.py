#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from tensorflow import keras
import scipy.io as sio
# from adabelief_tf import AdaBeliefOptimizer

# Parameters Setting
NUM_FEEDBACK_BITS = 378
CHANNEL_SHAPE_DIM1 = 24
CHANNEL_SHAPE_DIM2 = 16
CHANNEL_SHAPE_DIM3 = 2

num_quan_bits = 3

# Data Loading
mat = sio.loadmat('./channelData/H_4T4R.mat')
data = mat['H_4T4R']
data = data.astype('float32')
data = np.reshape(data, (len(data), CHANNEL_SHAPE_DIM1, CHANNEL_SHAPE_DIM2, CHANNEL_SHAPE_DIM3))


# # 定义一些工具函数

# In[2]:


import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def Num2Bit(Num, B):
    Num_ = Num.numpy()
    bit = (np.unpackbits(np.array(Num_, np.uint8), axis=1).reshape(-1, Num_.shape[1], 8)[:, :, (8-B):]).reshape(-1,Num_.shape[1] * B)
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
    x.set_shape((None, int(NUM_FEEDBACK_BITS/num_quan_bits)))
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


import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as L
# import tensorflow_addons as tfa




class SqueezeExciteLayer(L.Layer):
    def __init__(self, ratio, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ratio = ratio
        self.pool = None
        self.dense1 = None
        self.dense2 = None

    def build(self, input_shape):
        orig_size = input_shape[-1]
        squeeze_size = max(orig_size // self.ratio, 4)
        
        self.pool = L.GlobalAveragePooling2D()
        self.dense1 = L.Dense(squeeze_size, activation='relu')
        self.dense2 = L.Dense(orig_size, activation='sigmoid')
        
    def call(self, batch_input):
        x = self.pool(batch_input)
        x = self.dense1(x)
        x = self.dense2(x)
        x = tf.reshape(x, shape=(-1, 1, 1, batch_input.shape[-1]))
        return x * batch_input
    
    def get_config(self):
        cfg = super().get_config()
        cfg.update({'ratio': self.ratio})
        return cfg


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
    return {"QuantizationLayer":QuantizationLayer,"DeuantizationLayer":DeuantizationLayer, "SqueezeExciteLayer": SqueezeExciteLayer}

# Comliling
def NMSE_cuda_loss(y_true, y_pred):
    y_true = y_true-0.5
    y_pred = y_pred-0.5
    mse = tf.reduce_sum(tf.square(y_true-y_pred) , axis=(1,2,3))
    dominator = tf.reduce_sum(tf.square(y_true) , axis=(1,2,3))
    nmse = mse/dominator

    # ssim = nmse*tf.image.ssim(y_true, y_pred, 1, filter_size=11, filter_sigma=1.5, k1=0.01, k2=0.03)
    # nmse = nmse * tf.math.exp((nmse-0.1)*2)
    
    return tf.reduce_mean(nmse) 

def NMSE_cuda(y_true, y_pred):
    y_true = y_true-0.5
    y_pred = y_pred-0.5
    mse = tf.reduce_sum(tf.square(y_true-y_pred) , axis=(1,2,3))
    dominator = tf.reduce_sum(tf.square(y_true) , axis=(1,2,3))
    nmse = mse/dominator
    
    return nmse


def ssim_loss(y_true, y_pred):
  return tf.image.ssim(y_true, y_pred, 1, filter_size=11, filter_sigma=1.5, k1=0.01, k2=0.03)

class NMSE_metric(tf.keras.metrics.Metric):
    def __init__(self, name="NMSE", **kwargs):
        super(NMSE_metric, self).__init__(name=name, **kwargs)
        self.true_positives = self.add_weight(name="ctp", initializer="zeros")
        self.count = self.add_weight(name="ctp", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.true_positives.assign_add( tf.reduce_mean(NMSE_cuda(y_true, y_pred)) )
        # self.true_positives.assign( tf.reduce_mean(NMSE_cuda(y_true, y_pred)) )

        self.count.assign_add(1)
    def result(self):
        return self.true_positives/self.count
        # return self.true_positives

    def reset_states(self):
        # The state of the metric will be reset at the start of each epoch.
        self.true_positives.assign(0.0)
        self.count.assign(0.0)


# In[3]:


# Encoder and Decoder Function Defining

from tensorflow.keras.layers import Dense, Lambda
from tensorflow.keras.layers import Activation
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Conv2D, BatchNormalization, concatenate, add
from tensorflow.keras.regularizers import l2
import tensorflow.keras as K


class Rezero(layers.Layer):
    def __init__(self, *args, **kwargs):
        super(Rezero, self).__init__(*args, **kwargs)

    def build(self, input_shape):
        self.resweight = tf.Variable(0.3, trainable=True)

    def call(self, res, x):
        x = res + x * self.resweight
        return x

class Noise(layers.Layer):
    def __init__(self, *args, **kwargs):
        super(Noise, self).__init__(*args, **kwargs)

    def build(self, input_shape):
       pass

    def call(self, h):
        noise = tf.keras.backend.random_normal(shape=tf.shape(h))
        STD = tf.math.reduce_std(h, axis=0)

        return h + noise * STD * 0.005


def add_common_layers(y):
    y = keras.layers.BatchNormalization()(y)
    y = keras.layers.PReLU()(y)
    # y = layers.Activation(tf.nn.swish)(y)
    return y

def __grouped_convolution_block(input, grouped_channels, cardinality, strides, weight_decay=0):
    ''' Adds a grouped convolution block. It is an equivalent block from the paper
    Args:
        input: input tensor
        grouped_channels: grouped number of filters
        cardinality: cardinality factor describing the number of groups
        strides: performs strided convolution for downscaling if > 1
        weight_decay: weight decay term
    Returns: a keras tensor
    '''
    init = input
    channel_axis = -1

    group_list = []

    if cardinality == 1:
        # with cardinality 1, it is a standard convolution
        x = Conv2D(grouped_channels, (3, 3), padding='same', use_bias=False, strides=(strides, strides),
                   kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(init)
        x = BatchNormalization(axis=channel_axis)(x)
        x = layers.PReLU()(x)
        return x

    for c in range(cardinality):
        x = Lambda(lambda z: z[:, :, :, c * grouped_channels:(c + 1) * grouped_channels])(input)

        x = Conv2D(grouped_channels, (7, 7), padding='same', use_bias=False, strides=(strides, strides),
                   kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(x)

        group_list.append(x)

    group_merge = concatenate(group_list, axis=channel_axis)
    x = BatchNormalization(axis=channel_axis)(group_merge)
    x = layers.PReLU()(x)
    # x = layers.Activation(tf.nn.swish)(x)

    return x


def __bottleneck_block(input, filters=96, cardinality=16, strides=1, weight_decay=0):
    ''' Adds a bottleneck block
    Args:
        input: input tensor
        filters: number of output filters
        cardinality: cardinality factor described number of
            grouped convolutions
        strides: performs strided convolution for downsampling if > 1
        weight_decay: weight decay factor
    Returns: a keras tensor
    '''
    init = input

    grouped_channels = int(filters / cardinality)
    channel_axis = -1


    if init.shape[-1] != 2 * filters:
        init = Conv2D(filters * 2, (1, 1), padding='same', strides=(strides, strides),
                      use_bias=False, kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(init)
        init = BatchNormalization(axis=channel_axis)(init)

    x = Conv2D(filters, (1, 1), padding='same', use_bias=False,
               kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(input)
    x = BatchNormalization(axis=channel_axis)(x)
    x = layers.PReLU()(x)
    # x = layers.Activation(tf.nn.swish)(x)

    x = __grouped_convolution_block(x, grouped_channels, cardinality, strides, weight_decay)

    x = Conv2D(filters * 2, (1, 1), padding='same', use_bias=False, kernel_initializer='he_normal',
               kernel_regularizer=l2(weight_decay))(x)
    x = BatchNormalization(axis=channel_axis)(x)
    x = SqueezeExciteLayer(16)(x)


    x = add([init, x])
    x = layers.PReLU()(x)

    return x



def ACR_DEC(x, k=8):
    x = __bottleneck_block(x)

    return x

def ACR_ENC(x):
    res = x
    input_ch = x.shape[-1]

    x = layers.Conv2D(input_ch, kernel_size=(1,9), strides=1, padding='same', data_format='channels_last', use_bias=False)(x)
    x = add_common_layers(x)

    x = layers.Conv2D(input_ch, kernel_size=(9,1), strides=1, padding='same', data_format='channels_last', use_bias=False)(x)
    x = layers.BatchNormalization()(x)

    x = SqueezeExciteLayer(16)(x)

    x = layers.PReLU()(x+res)

    return x

def ACR_ENC2(x):
    res = x
    input_ch = x.shape[-1]

    x = layers.Conv2D(input_ch, kernel_size=(1,3), strides=1, padding='same', data_format='channels_last', use_bias=False)(x)
    x = add_common_layers(x)

    x = layers.Conv2D(input_ch, kernel_size=(3,1), strides=1, padding='same', data_format='channels_last', use_bias=False)(x)
    x = layers.BatchNormalization()(x)

    x = SqueezeExciteLayer(16)(x)

    x = layers.PReLU()(x+res)

    return x


# # 模型架构

# ### Encoder部分的代码如下

# In[4]:


# NUM_FEEDBACK_BITS = 378
# CHANNEL_SHAPE_DIM1 = 24
# CHANNEL_SHAPE_DIM2 = 16
# CHANNEL_SHAPE_DIM3 = 2
# num_quan_bits = 3


def Encoder(enc_input,num_feedback_bits):
    num_quan_bits = 3

    h = enc_input-0.5
    h0 = layers.BatchNormalization()(h)

	# 使用这个版本的模型可能会出现过拟合，这里的结构仅供参考
	
    h = layers.concatenate([h0[:,:,:,0], h0[:,:,:,1]], axis=-1)

    for i in range(1,5):
      h1 = layers.Conv1D(filters=512//(2**i), kernel_size=1, padding='same', activation='linear')(h)
      h1 = add_common_layers(h1)
      h3 = layers.Conv1D(filters=512//(2**i), kernel_size=3, padding='same', activation='linear')(h)
      h3 = add_common_layers(h3)
      h5 = layers.Conv1D(filters=512//(2**i), kernel_size=5, padding='same', activation='linear')(h)
      h5 = add_common_layers(h5)

      h = layers.concatenate([h1, h3, h5], axis=-1)

    h = layers.Conv1D(filters=16, kernel_size=1, padding='same', activation='linear')(h)
    h = add_common_layers(h)

    h_last = layers.Flatten()(h)


    conv1 = layers.Conv2D(128, kernel_size=3, strides=1, padding='same', data_format='channels_last', use_bias=False, kernel_initializer='he_normal')
    h = conv1(h0)
    h = add_common_layers(h)


    h = ACR_ENC(h)
    h = ACR_ENC(h)
    h = ACR_ENC2(h)
    h = ACR_ENC2(h)


    conv2 = layers.Conv2D(256, kernel_size=3, strides=1, padding='same', data_format='channels_last', use_bias=False, kernel_initializer='he_normal')
    h = conv2(h)
    h = add_common_layers(h)



    h = ACR_ENC(h)
    h = ACR_ENC(h)


    conv3 = layers.Conv2D(512, kernel_size=3, strides=1, padding='same', data_format='channels_last', use_bias=False, kernel_initializer='he_normal')
    h = conv3(h)
    h = add_common_layers(h)

    h = layers.Conv2D(8, kernel_size=3, strides=1, padding='same', data_format='channels_last', use_bias=False, kernel_initializer='he_normal')(h)
    h = layers.BatchNormalization()(h)

    h = layers.Flatten()(h)

    h = layers.concatenate([h, h_last], axis=-1)


    h = layers.Dense(units=int(num_feedback_bits / num_quan_bits), activation='linear', kernel_initializer='he_normal')(h)
    h = layers.BatchNormalization()(h)
    h = layers.Activation('sigmoid', name='input_feat')(h)

    enc_output = QuantizationLayer(num_quan_bits)(h)
    return enc_output


# ### Decoder部分的代码如下，这里参考了三种常见的CNN架构：
#  - 残差链接
#  - 分组卷积
#  - SE block

# In[5]:


def Decoder(dec_input,num_feedback_bits):
    num_quan_bits = 3
#     print('xx1x', dec_input.shape)
    h = DeuantizationLayer(num_quan_bits)(dec_input)
#     print('xx2x', h.shape)
    h = tf.keras.layers.Reshape((int(num_feedback_bits/num_quan_bits), ))(h)
    h = tf.math.log((1/h-1))

    h = layers.Dense(24*16*2, activation='linear')(h)
    h = tf.keras.layers.Reshape((24,16,2))(h)
    h = layers.BatchNormalization()(h)

    h = layers.Conv2D(256, kernel_size=3, strides=1, padding='same', data_format='channels_last', use_bias=False)(h)
    h = layers.BatchNormalization()(h)

    for i in range(9):
      res = h
      num_ch = h.shape[-1]
      h = layers.Conv2D(num_ch*2, kernel_size=(1, 9), padding='same', data_format='channels_last', use_bias=False)(res)
      h = add_common_layers(h)
      h = layers.Conv2D(num_ch, kernel_size=(3, 3), padding='same', data_format='channels_last', use_bias=False)(h)
      h = layers.BatchNormalization()(h)
      
      h = res + h*(0.7**(i+1))
      h = layers.PReLU()(h)

      h = ACR_DEC(h)
      
      
    h = layers.Conv2D(2, kernel_size=(3, 3), padding='same', data_format='channels_last')(h)

    dec_output = h/10000+0.5

    return dec_output


# # 模型训练
# 这里我们采用了指数衰减式的学习率方案，每一轮衰减0.1， 初始学习率设置为0.002。验证集取最后的0.05数据。

# In[ ]:


encInput = keras.Input(shape=(CHANNEL_SHAPE_DIM1, CHANNEL_SHAPE_DIM2, CHANNEL_SHAPE_DIM3))
encOutput = Encoder(encInput, NUM_FEEDBACK_BITS)
encModel = keras.Model(inputs=encInput, outputs=encOutput, name='Encoder')

decInput = keras.Input(shape=(NUM_FEEDBACK_BITS,))
decOutput = Decoder(decInput, NUM_FEEDBACK_BITS)
decModel = keras.Model(inputs=decInput, outputs=decOutput, name="Decoder")


autoencoderInput = keras.Input(shape=(CHANNEL_SHAPE_DIM1, CHANNEL_SHAPE_DIM2, CHANNEL_SHAPE_DIM3))
h = encModel(autoencoderInput)
autoencoderOutput = decModel(h)

autoencoderModel = keras.Model(inputs=autoencoderInput, outputs=autoencoderOutput, name='Autoencoder')


def scheduler(epoch, lr):
    return max(lr * 0.9, 5e-6)
callback = tf.keras.callbacks.LearningRateScheduler(scheduler)


class CustomCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        encModel.save('./save/models/encoder%s.h5'%epoch)
        decModel.save('./save/models/decoder%s.h5'%epoch)
cb2 = CustomCallback()

def NMSE_cuda_loss(y_true, y_pred):
    y_true = y_true-0.5
    y_pred = y_pred-0.5
    mse = tf.reduce_sum(tf.square(y_true-y_pred) , axis=(1,2,3))
    dominator = tf.reduce_sum(tf.square(y_true) , axis=(1,2,3))
    nmse = mse/dominator
 
    return tf.reduce_mean(nmse) 


loss = NMSE_cuda_loss
# optimizer = AdaBeliefOptimizer(learning_rate=0.002, epsilon=1e-14, rectify=False, print_change_log=False)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.002, epsilon=1e-14)
autoencoderModel.compile(optimizer=optimizer, loss=loss, metrics=[NMSE_metric()])
print(autoencoderModel.summary())

# encModel.save('./modelSubmit/encoder.h5')
# decModel.save('./modelSubmit/decoder.h5')

autoencoderModel.fit(x=data, y=data, 
            batch_size=128, 
            epochs=100, 
            verbose=1, 
            validation_split=0.05, 
            shuffle=True, 
            callbacks=[callback, cb2]
            )


# In[ ]:




'''md
方案亮点
这里的方案主要亮点在于残差层之间的衰减系数，理论上这一步没有必要，但是对于使用梯度传播的神经网络优化时，这一操作却能取得不错的效果。

学习率衰减可以加快模型收敛，但是过大的衰减速率会导致欠拟合。

模型微调
通过重复利用模型权重可以节省不少的训练时间。具体的实现代码如下

```python
NUM_FEEDBACK_BITS = 378
CHANNEL_SHAPE_DIM1 = 24
CHANNEL_SHAPE_DIM2 = 16
CHANNEL_SHAPE_DIM3 = 2
num_quan_bits = 3

subencModel = keras.Model(inputs=encModel.input, outputs=encModel.layers[-2].output, name='subEncoder')

encInput = keras.Input(shape=(CHANNEL_SHAPE_DIM1, CHANNEL_SHAPE_DIM2, CHANNEL_SHAPE_DIM3))
h0 = subencModel(encInput)

N_train = 12
if N_train>0:
  h = layers.Dense(units=int(N_train / num_quan_bits), activation='linear', kernel_initializer='he_normal')(h0)
  h = layers.Activation('sigmoid', name='input_feat')(h)

  alpha = tf.Variable(0.1, trainable=True)
  h = h0[:,int((NUM_FEEDBACK_BITS-N_train)/3):int((NUM_FEEDBACK_BITS)/3)] * (1-alpha) + h * alpha


  h = layers.concatenate([h0[:,:int((NUM_FEEDBACK_BITS-N_train)/3)], h], axis=-1)
else:
  h = h0[:,:(NUM_FEEDBACK_BITS//3)]

encOutput = QuantizationLayer(num_quan_bits)(h)

encModel0 = keras.Model(inputs=encInput, outputs=encOutput, name='Encoder0')


decInput = keras.Input(shape=(NUM_FEEDBACK_BITS,))

decInput = decInput

h = DeuantizationLayer(num_quan_bits)(decInput)
h0 = tf.keras.layers.Reshape((int(NUM_FEEDBACK_BITS/num_quan_bits), ))(h)

h = layers.Dense(units=int((378-NUM_FEEDBACK_BITS) / num_quan_bits), activation='linear', kernel_initializer='he_normal')(h0)
h = tf.keras.layers.Reshape((int((378-NUM_FEEDBACK_BITS)/num_quan_bits), ))(h)
h = layers.Activation('sigmoid')(h)

h_ = layers.Dense(units=int((378) / num_quan_bits), activation='linear', kernel_initializer='he_normal')(h0)
h_ = tf.keras.layers.Reshape((378//num_quan_bits, ))(h_)
h_ = layers.Activation('sigmoid')(h_)

h = layers.concatenate([h0, h], axis=-1)

alpha = tf.Variable(0.001, trainable=True)
h = h * (1-alpha) + h_ * alpha

h = QuantizationLayer(num_quan_bits)(h)

decOutput = decModel(h)
decModel0 = keras.Model(inputs=decInput, outputs=decOutput, name="Decoder0")


autoencoderInput = keras.Input(shape=(CHANNEL_SHAPE_DIM1, CHANNEL_SHAPE_DIM2, CHANNEL_SHAPE_DIM3))
h = encModel0(autoencoderInput)
autoencoderOutput = decModel0(h)

autoencoderModel = keras.Model(inputs=autoencoderInput, outputs=autoencoderOutput, name='Autoencoder')

```
'''




