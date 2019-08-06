# coding=utf-8
import os
import re
import sys
import pandas as pd
import numpy as np
import gensim
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import tensorflow as tf
import keras
from keras.layers import *
from keras.models import *
from keras.optimizers import *
from keras.callbacks import *
from keras.preprocessing import text, sequence
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import  accuracy_score
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')
import os
import gc
import random
from keras.engine.topology import Layer
from m1 import BOW

maxlen = 100 # 每句话的固定长度(截断或者补全)
batch_size = 64
embedding_dim = 300
epochs = 10

comments = [['same', 'coffee', 'shop', 'my', 'memory', 'of', 'it', 'is'],[]]


# 训练词向量---------------------------------------
w2v_model = Word2Vec(comments,size=embedding_dim, min_count=5, workers=10)

# 构造embedding字典
bow = BOW(comments.tolist(), min_count=5, maxlen=maxlen)
vocab_size = len(bow.word2idx)

embedding_matrix = np.zeros((vocab_size+1,300))
for key, value in bow.word2idx.items():
    if key in w2v_model.wv.vocab: # Word2Vec训练得到的的实例需要word2vec.wv.vocab
        embedding_matrix[value] = w2v_model.wv[key]
    else:
        embedding_matrix[value] = [0] * embedding_dim

# 构建数据集-------------------------------------
X = copy.deepcopy(bow.doc2num[:159571])
# 训练集和验证集划分 4:1
kf = KFold(n_splits=5)
for train_idx, val_idx in kf.split(df.label):
    pass
train_X, train_y = X[train_idx],  np.array(df.label)[train_idx]
val_X, val_y = X[val_idx],  np.array(df.label)[val_idx]
# 之前的comments数据是拼接起来的，159571行以后是测试集
test = copy.deepcopy(bow.doc2num[159571:])

# 用keras构建bigru模型-------------------------------------
def bi_gru_model(maxlen, embeddings_weight, class_num):
    content = Input(shape=(maxlen,), dtype='int32')
    embedding = Embedding(
        name="word_embedding",
        input_dim=embeddings_weight.shape[0],
        weights=[embeddings_weight],
        output_dim=embeddings_weight.shape[1],
        trainable=False)

    x = SpatialDropout1D(0.2)(embedding(content))

    x = Bidirectional(GRU(200, return_sequences=True))(x)
    x = Bidirectional(GRU(200, return_sequences=True))(x)

    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)

    conc = concatenate([avg_pool, max_pool])

    x = Dropout(0.2)(Activation(activation="relu")(BatchNormalization()(Dense(1000)(conc))))
    x = Activation(activation="relu")(BatchNormalization()(Dense(500)(x)))
    output = Dense(class_num, activation="sigmoid")(x)

    model = Model(inputs=content, outputs=output)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# 训练
gru_model = bi_gru_model(maxlen, embedding_matrix, 1)
for i in range(1):
    gru_model.fit(train_X, train_y, epochs=10, batch_size=128, validation_data=(val_X, val_y))