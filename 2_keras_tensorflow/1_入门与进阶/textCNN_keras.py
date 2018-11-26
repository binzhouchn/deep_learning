# coding=utf-8
import numpy as np
import pandas as pd
import re
import copy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import gensim
from gensim.models.word2vec import Word2Vec
from keras import Input, Model
from keras.layers import Embedding, Dense, Conv1D, GlobalMaxPooling1D, Concatenate, Dropout
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

# 用keras构建textCNN模型 version1（这样写比较好一点）
class TextCNN(object):
    def __init__(self, maxlen, max_features, embedding_dim,
                 weights=None,
                 trainable=False,
                 class_num=1,
                 last_activation='sigmoid',
                 optimizer='adam',
                 loss='binary_crossentropy',
                 metrics=None,
                 batch_size=64,
                 epochs=10,
                 callbacks=None):
        self.maxlen = maxlen
        self.max_features = max_features
        self.embedding_dim = embedding_dim
        self.weights = weights  # weight list
        self.trainable = trainable
        self.class_num = class_num
        self.last_activation = last_activation
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics  # metric list, calculated at end of a batch, always useless
        self.batch_size = batch_size
        self.epochs = epochs
        self.callbacks = callbacks  # callbacks list, e.g., model checkpoint, early stopping, customed epoch-level metric...

    def fit(self, x_train, y_train, x_val, y_val):
        input = Input((self.maxlen,))
        embedding = Embedding(self.max_features, self.embedding_dim,
                              input_length=self.maxlen,
                              weights=self.weights,
                              trainable=self.trainable)(input)
        convs = []
        for kernel_size in [3, 4, 5]:
            c = Conv1D(128, kernel_size, activation='relu')(embedding)
            c = GlobalMaxPooling1D()(c)
            convs.append(c)
        x = Concatenate()(convs)
        x = Dropout(0.5)(x)
        output = Dense(self.class_num, activation=self.last_activation)(x)
        self.model = Model(inputs=input, outputs=output)

        self.model.compile(self.optimizer, loss=self.loss, metrics=self.metrics)

        self.model.fit(x_train, y_train,
                       validation_data=(x_val, y_val),
                       batch_size=self.batch_size,
                       epochs=self.epochs,
                       callbacks=self.callbacks)

        return self.model

    def predict(self, x):
        return self.model.predict(x)

model = TextCNN(maxlen, embedding_matrix.shape[0], embedding_matrix.shape[1],
                weights=[embedding_matrix],
                trainable=False,
                optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy'],
                batch_size=batch_size,
                epochs=epochs)

model.fit(train_X, train_y, val_X, val_y)