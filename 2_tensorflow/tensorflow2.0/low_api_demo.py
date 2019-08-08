# coding: utf-8
# File: low_api_demo.py
# Author: zhoubin
# Date: 20190808
# 使用低级api训练（非tf.keras）

import tensorflow as tf

###################1. 定义模型和损失函数###################
class Model(object):
    def __init__(self):
        # 初始化变量
        self.W = tf.Variable(5.0)
        self.b = tf.Variable(0.0)

    def __call__(self, x):
        return self.W * x + self.b
def loss(predicted_y, true_y):
    return tf.reduce_mean(tf.square(predicted_y - true_y))