[tensorflow官网](https://tensorflow.google.cn/)<br>
[keras英文官网](https://keras.io/)

# 目录

[**1. tensorflow1.x_withKeras**](tensorflow1.x_withKeras)

[**2. tensorflow2.0**](tensorflow2.0)

## tensorflow2.0笔记

[**mnist_demo**](#mnist_demo)

[**用tf.keras构建自己的网络层**](#用tf_keras构建自己的网络层)

[**保持序列模型和函数模型**](#保持序列模型和函数模型)

---

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




