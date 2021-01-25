#=======================================================================================================================
#=======================================================================================================================
# Train
import numpy as np
from tensorflow import keras
import scipy.io as sio
import os
import random
#=======================================================================================================================
#=======================================================================================================================
# Parameters Setting
NUM_FEEDBACK_BITS = 512
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
#=======================================================================================================================
#=======================================================================================================================
# Data Loading and split
mat = sio.loadmat('./channelData/H_4T4R.mat')
data = mat['H_4T4R']
data = data.astype('float32')
data = np.reshape(data, (len(data), CHANNEL_SHAPE_DIM1, CHANNEL_SHAPE_DIM2, CHANNEL_SHAPE_DIM3))
split = int(data.shape[0] * 0.8)
data_train, data_val = data[:split], data[split:]
#checkpointer for val dataset
checkpointer = CheckPointer(data_val)

#=======================================================================================================================
#=======================================================================================================================
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
autoencoderModel.compile(optimizer='adam', loss='mse', metrics=[NMSE_tf()])
print(autoencoderModel.summary())
#=======================================================================================================================
#=======================================================================================================================
# Model Training
autoencoderModel.fit(x=data_train, y=data_train, batch_size=128, \
                     epochs=4,shuffle=True, verbose=1, callbacks=[checkpointer])
#=======================================================================================================================
#=======================================================================================================================

