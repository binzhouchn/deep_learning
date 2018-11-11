# tensorflow和keras入门

[**1. 吴恩达tensorflow入门代码(用tensorflow求导)**](1.tensorflow_andrewNg.ipynb)

[**2. 查看tensorflow或keras的张量值**](#查看tensorflow或keras的张量值)

[**3. keras中加载预训练的embedding**](#keras中加载预训练的embedding)

[**4. tensorflow和keras保存读取**](#tensorflow和keras保存读取)

### 查看tensorflow或keras的张量值

```python
from keras.layers import K
x = K.random_normal(shape = (64,100,256))
w = K.random_normal(shape = (1,256,160))
res = K.conv1d(x,w)

# run tf
init=tf.global_variables_initializer() # 初始化（必须）
with tf.Session() as sess:
    sess.run(init)
    cc = sess.run(x)
    dd = sess.run(res)
    print(cc)
    print('------------------------------')
    print(dd)
```

### keras中加载预训练的embedding

```python
from keras.initializers import Constant
Embedding(vocab_size + 1,
            EMBEDDING_DIM,
            embeddings_initializer=Constant(embedding_matrix),
            input_length=maxlen,
            trainable=False)
```
[参考代码](https://github.com/keras-team/keras/blob/master/examples/pretrained_word_embeddings.py)

### tensorflow和keras保存读取

[tensorflow Saver 保存读取](https://morvanzhou.github.io/tutorials/machine-learning/tensorflow/5-06-save/)

[keras Save&reload 保存提取](https://morvanzhou.github.io/tutorials/machine-learning/keras/3-1-save/)

