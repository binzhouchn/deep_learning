# tf2学习

版本tensorflow2.1.0
[学习github网址](https://github.com/lyhue1991/eat_tensorflow2_in_30_days)<br>

## 1.安装

`docker pull binzhouchn/tensorflow:2.1.0-cuda10.1-cudnn7`

## 2.基础

Tensorflow一般使用梯度磁带tf.GradientTape来记录正向运算过程，然后反播磁带自动得到梯度值。<br>
```python
# 一阶导
x = tf.Variable(0.0,name = "x",dtype = tf.float32)
with tf.GradientTape() as tape:
    y = a*tf.pow(x,2) + b*x + c
tape.gradient(y,x)

# 二阶导
with tf.GradientTape() as tape2:
    with tf.GradientTape() as tape1:   
        y = a*tf.pow(x,2) + b*x + c
    dy_dx = tape1.gradient(y,x)   
dy2_dx2 = tape2.gradient(dy_dx,x)

```

## 3.建模

[文本数据建模流程范例](https://github.com/lyhue1991/eat_tensorflow2_in_30_days/blob/master/1-3,%E6%96%87%E6%9C%AC%E6%95%B0%E6%8D%AE%E5%BB%BA%E6%A8%A1%E6%B5%81%E7%A8%8B%E8%8C%83%E4%BE%8B.md)<br>












### 注

[使用spark-scala调用tensorflow2.0训练好的模型](https://blog.csdn.net/zimiao552147572/article/details/105330740)<br>