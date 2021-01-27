# =======================================================================================================================
# =======================================================================================================================
# Train
import numpy as np
from tensorflow import keras
from modelDesign import *
import scipy.io as sio
import os
import random

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    gpu = gpus[0]  # 如果有多个GPU，仅使用第1个GPU
    tf.config.experimental.set_memory_growth(gpu, True)  # 显存需要多少用多少，而不是一次占满
    tf.config.set_visible_devices([gpu], "GPU")
# =======================================================================================================================
# =======================================================================================================================
# Parameters Setting
NUM_FEEDBACK_BITS = 468
CHANNEL_SHAPE_DIM1 = 24
CHANNEL_SHAPE_DIM2 = 16
CHANNEL_SHAPE_DIM3 = 2


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


SEED = 42
seed_everything(SEED)
# =======================================================================================================================
# =======================================================================================================================
# Data Loading and split
mat = sio.loadmat('./channelData/H_4T4R.mat')
data = mat['H_4T4R']
data = data.astype('float32')
data = np.reshape(data, (len(data), CHANNEL_SHAPE_DIM1, CHANNEL_SHAPE_DIM2, CHANNEL_SHAPE_DIM3))
split = int(data.shape[0] * 0.8)
data_train, data_val = data[:split], data[split:]
# checkpointer for val dataset
checkpointer = CheckPointer(data_val)#验证集添加到checkpoint，每次满足best val的时候保存模型

# =======================================================================================================================
# =======================================================================================================================
# Model Constructing
# Encoder
encInput = keras.Input(shape=(CHANNEL_SHAPE_DIM1, CHANNEL_SHAPE_DIM2, CHANNEL_SHAPE_DIM3))
encOutput = Encoder(encInput, NUM_FEEDBACK_BITS)
encModel = keras.Model(inputs=encInput, outputs=encOutput, name='Encoder')
# Decoder
decInput = keras.Input(shape=(NUM_FEEDBACK_BITS,))
decOutput = Decoder(decInput, NUM_FEEDBACK_BITS)
decModel = keras.Model(inputs=decInput, outputs=decOutput, name="Decoder")
# Autoencoder
autoencoderInput = keras.Input(shape=(CHANNEL_SHAPE_DIM1, CHANNEL_SHAPE_DIM2, CHANNEL_SHAPE_DIM3))
autoencoderOutput = decModel(encModel(autoencoderInput))
autoencoderModel = keras.Model(inputs=autoencoderInput, outputs=autoencoderOutput, name='Autoencoder')
# Comliling
optim = tf.keras.optimizers.Adam(learning_rate=2e-3)
autoencoderModel.compile(optimizer=optim, loss='mse', metrics=[NMSE_tf()])
print(autoencoderModel.summary())
# =======================================================================================================================
# =======================================================================================================================
# Model Training
autoencoderModel.fit(x=data_train, y=data_train, batch_size=128, \
                     epochs=100, shuffle=True, verbose=1, callbacks=[checkpointer])
# =======================================================================================================================
# =======================================================================================================================
