# tf2学习

版本tensorflow2.1.0
[学习github网址](https://github.com/lyhue1991/eat_tensorflow2_in_30_days)<br>

## 目录

 - [**1. 安装**](#安装)
 - [**2. 基础**](#基础)
 - [**3. 建模**](#建模)


### 安装

`docker pull binzhouchn/tensorflow:2.1.0-cuda10.1-cudnn7`

### 基础

1. 自变量转换成tf.float32<br>
```python
x = tf.cast(x, tf.float32)
```

2. Tensorflow一般使用梯度磁带tf.GradientTape来记录正向运算过程，然后反播磁带自动得到梯度值。<br>
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

3. 利用梯度磁带和优化器求最小值<br>

[自动微分详见](https://github.com/lyhue1991/eat_tensorflow2_in_30_days/blob/master/2-3,%E8%87%AA%E5%8A%A8%E5%BE%AE%E5%88%86%E6%9C%BA%E5%88%B6.md)<br>
```python
# 求f(x) = a*x**2 + b*x + c的最小值
# 使用optimizer.apply_gradients

x = tf.Variable(0.0,name = "x",dtype = tf.float32)
a = tf.constant(1.0)
b = tf.constant(-2.0)
c = tf.constant(1.0)

optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
for _ in range(1000):
    with tf.GradientTape() as tape:
        y = a*tf.pow(x,2) + b*x + c
    dy_dx = tape.gradient(y,x)
    optimizer.apply_gradients(grads_and_vars=[(dy_dx,x)])
    
tf.print("y =",y,"; x =",x)
```

4. 取切片数据<br>
```python
x = tf.Variable([1,2,3,4,5,6])
slice_idx = tf.constant([0,3,5])
tf.gather(x, slice_idx)
#得到<tf.Tensor: shape=(3,), dtype=int32, numpy=array([1, 4, 6], dtype=int32)>
```

5. 矩阵乘法<br>

用tf.matmul或者@

6. tf2低阶api - 张量的结构操作<br>

 - 一，创建张量
 - 二 ，索引切片
 - 三，维度变换
 - 四，合并分割

[链接，和numpy很类似](https://github.com/lyhue1991/eat_tensorflow2_in_30_days/blob/master/4-1,%E5%BC%A0%E9%87%8F%E7%9A%84%E7%BB%93%E6%9E%84%E6%93%8D%E4%BD%9C.md)

7. xxx


### 建模

[文本数据建模流程范例](https://github.com/lyhue1991/eat_tensorflow2_in_30_days/blob/master/1-3,%E6%96%87%E6%9C%AC%E6%95%B0%E6%8D%AE%E5%BB%BA%E6%A8%A1%E6%B5%81%E7%A8%8B%E8%8C%83%E4%BE%8B.md)<br>












### 注

[使用spark-scala调用tensorflow2.0训练好的模型](https://blog.csdn.net/zimiao552147572/article/details/105330740)<br>