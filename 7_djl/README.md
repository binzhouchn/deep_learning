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
或者自己搭建Dockerfile<br>
```shell
FROM ubuntu:18.04

RUN apt-get update || true
RUN apt-get install -y openjdk-11-jdk-headless
RUN apt-get install -y python3-pip git
RUN pip3 install jupyter
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y locales \
    && sed -i -e 's/# en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/' /etc/locale.gen \
    && dpkg-reconfigure --frontend=noninteractive locales \
    && update-locale LANG=en_US.UTF-8
RUN apt-get install -y curl

RUN git clone https://github.com/frankfliu/IJava.git
RUN cd IJava/ && ./gradlew installKernel && cd .. && rm -rf IJava/
RUN rm -rf ~/.gradle

WORKDIR /home/jupyter

ENV LANG en_US.UTF-8
ENV LC_ALL en_US.UTF-8

EXPOSE 8888
ENTRYPOINT ["jupyter", "notebook", "--ip=0.0.0.0", "--no-browser", "--allow-root", "--NotebookApp.token=''",  "--NotebookApp.password=''"]
```
```shell
cd jupyter
docker build -t deepjavalibrary/jupyter .
```

## 一些简单的模型训练与预测

[手写数字](1.mnist_demo_java.ipynb)<br>
[物品检测](2.object_detection_with_model_zoo.ipynb)<br>


