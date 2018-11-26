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

# 训练词向量
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

# 用keras构建textCNN模型 version1（这样写比较好一点）
