[tensorflow官网](https://tensorflow.google.cn/)<br>
[keras英文官网](https://keras.io/)

# 目录

[**1. tensorflow1.x_withKeras**](tensorflow1.x_withKeras)

[**2. tensorflow2.0**](tensorflow2.0)

---

## tensorflow2.0笔记

[**mnist_demo**](#mnist_demo)

[**用tf.keras构建自己的网络层**](#用tf_keras构建自己的网络层)

[**保持序列模型和函数模型**](#保持序列模型和函数模型)

[**结构化数据分类(一般csv文件数据特征处理)**](#结构化数据分类)

[**mlp及深度学习常见技巧**](#mlp及深度学习常见技巧)

[**使用低级api训练(非tf.keras)**](#使用低级api训练)

[**Transformer**](#Transformer)



### mnist_demo

见tensorflow2.0目录下代码

### 用tf_keras构建自己的网络层

见tensorflow2.0目录下代码

### 保持序列模型和函数模型

1.1 保存模型参数（推荐）
```python
# 保存
model.save_weights('my_model_weights', save_format='tf')
# 读取
new_model = get_model() # 之前设计好的模型结构
new_model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=keras.optimizers.RMSprop())
new_model.load_weights('my_model_weights')
```

1.2 保持全模型(子类模型的结构无法保存和序列化，只能保持参数)<br>
可以对整个模型进行保存，其保持的内容包括：<br>

 - 该模型的架构
 - 模型的权重（在训练期间学到的）
 - 模型的训练配置（你传递给编译的），如果有的话
 - 优化器及其状态（如果有的话）（这使您可以从中断的地方重新启动训练）
 
```python
import numpy as np
model.save('the_save_model.h5')
new_model = keras.models.load_model('the_save_model.h5')
new_prediction = new_model.predict(x_test)
np.testing.assert_allclose(predictions, new_prediction, atol=1e-6) # 预测结果一样
```

### 结构化数据分类

[分类](https://blog.csdn.net/qq_31456593/article/details/88777343)<br>
[回归](https://blog.csdn.net/qq_31456593/article/details/88778647)<br>

见代码tensorflow2.0/5.csv_binary_classify.py

### mlp及深度学习常见技巧

[以mlp为基础模型，然后介绍一些深度学习常见技巧， 如：
权重初始化， 激活函数， 优化器， 批规范化， dropout，模型集成](https://blog.csdn.net/qq_31456593/article/details/88915982)

### 使用低级api训练

使用Tensor， Variable和GradientTape这些简单的要是，就可以构建一个简单的模型。步骤如下：

[链接](https://blog.csdn.net/qq_31456593/article/details/95040964)

见代码tensorflow2.0/6.low_api_demo.py

### GAN

[链接](https://blog.csdn.net/qq_31456593/article/details/88991068)

见代码tensorflow2.0/002-DCGAN.ipynb

### Transformer

[这篇讲的确实详细非常好](https://blog.csdn.net/qq_31456593/article/details/89923913)<br>

已讲网页保存至tensorflow2.0/files，还有001-Transformer.ipynb也在files中