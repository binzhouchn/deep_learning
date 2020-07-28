# Deep Java Library (DJL)

## 1.安装Java Jupyter Kernel

要求jdk11及以上，maven3.6.3及以上<br>
```shell
java --list-modules | grep "jdk.jshell"

> jdk.jshell@12.0.1
```
```shell
git clone https://github.com/frankfliu/IJava.git
cd IJava/
./gradlew installKernel
```
然后启动jupyter notebook即可，选java kernel的notebook

## 手写数字训练与预测


[ipynb代码](mnist_demo_java.ipynb)

