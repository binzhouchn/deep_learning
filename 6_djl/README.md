# Deep Java Library (DJL)

## 1.安装Java Jupyter Kernel

【方式一】

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

【方式二】

```shell
docker pull deepjavalibrary/jupyter
mkdir jupyter
cd jupyter
docker run -itd -p 127.0.0.1:8888:8888 -v $PWD:/home/jupyter deepjavalibrary/jupyter
```

## 一些简单的模型训练与预测

[手写数字](1.mnist_demo_java.ipynb)<br>
[物品检测](2.object_detection_with_model_zoo.ipynb)<br>


