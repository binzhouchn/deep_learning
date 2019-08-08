# coding: utf-8
# File: low_api_demo.py
# Author: zhoubin
# Date: 20190808
# 使用低级api训练（非tf.keras）

import tensorflow as tf
import matplotlib.pyplot as plt

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

###################2. 构造数据（回归）###################
TRUE_W = 3.0
TRUE_b = 2.0
num = 1000
# 随机输入
inputs = tf.random.normal(shape=[num])
# 随机噪音
noise = tf.random.normal(shape=[num])
# 构造数据
outputs = TRUE_W * inputs + TRUE_b + noise

###################3. 训练###################
def train(model, inputs, outputs, learning_rate):
    # 记录loss计算过程
    with tf.GradientTape() as t: #自动求导
        current_loss = loss(model(inputs), outputs)
        # 对W，b求导
        dW, db = t.gradient(current_loss, [model.W, model.b])
        # 减去梯度×学习率
        model.W.assign_sub(dW*learning_rate)
        model.b.assign_sub(db*learning_rate)

model= Model()
# 收集W，b画图
Ws, bs = [], []
for epoch in range(10):
    Ws.append(model.W.numpy())
    bs.append(model.b.numpy())
    # 计算loss
    current_loss = loss(model(inputs), outputs)
    train(model, inputs, outputs, learning_rate=0.1)
    print('Epoch %2d: W=%1.2f b=%1.2f, loss=%2.5f' %
        (epoch, Ws[-1], bs[-1], current_loss))
# 画图
# Let's plot it all
epochs = range(10)
plt.plot(epochs, Ws, 'r',
         epochs, bs, 'b')
plt.plot([TRUE_W] * len(epochs), 'r--',
         [TRUE_b] * len(epochs), 'b--')
plt.legend(['W', 'b', 'true W', 'true_b'])
plt.show()