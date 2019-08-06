# coding: utf-8
# File: myLayer.py
# Author: zhoubin
# Date: 20190806
# 用tf.keras构建自己的网络层

# original
class MyLayer(layers.Layer):
    def __init__(self, input_dim=32, unit=32):
        super(MyLayer, self).__init__()
        self.weight = self.add_weight(shape=(input_dim, unit),
                                      initializer=keras.initializers.RandomNormal(),
                                      trainable=True)
        self.bias = self.add_weight(shape=(unit,),
                                    initializer=keras.initializers.Zeros(),
                                    trainable=True)

    def call(self, inputs):
        return tf.matmul(inputs, self.weight) + self.bias


x = tf.ones((3, 5))
my_layer = MyLayer(5, 4)
out = my_layer(x)
print(out)

# 当定义网络时不知道网络的维度是可以重写build()函数，用获得的shape构建网络
class MyLayer(layers.Layer):
    def __init__(self, unit=32):
        super(MyLayer, self).__init__()
        self.unit = unit

    def build(self, input_shape):
        self.weight = self.add_weight(shape=(input_shape[-1], self.unit),
                                      initializer=keras.initializers.RandomNormal(),
                                      trainable=True)
        self.bias = self.add_weight(shape=(self.unit,),
                                    initializer=keras.initializers.Zeros(),
                                    trainable=True)

    def call(self, inputs):
        return tf.matmul(inputs, self.weight) + self.bias


my_layer = MyLayer(3)
x = tf.ones((3, 5))
out = my_layer(x)
print(out)
my_layer = MyLayer(3)

