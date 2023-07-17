【已跑通！】

1.环境准备
 
 - 两台服务器（确保服务器之间是通的，并且关闭防火墙sudo ufw status查看状态），每台2张V100；*.*.72.6 ubuntu16.04lts（主）, *.*.72.7 ubuntu22.04lts（从）或者*.*.72.5 ubuntu16.04lts（从）
 - 以.6和.5为例，这两台都是ubuntu16.04lts, cuda11.3, cudnn8.4.0, cudatoolkit11.3
 - Anaconda3-2022.05-Linux-x86_64.sh
 - pip install torch==1.10.1+cu113 torchvision==0.11.2+cu113 -f https://download.pytorch.org/whl/torch_stable.html -i https://pypi.tuna.tsinghua.edu.cn/simple some-package
 或者conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge
 - transformers==4.23.1; tensorboard==2.13.0
 - tensorflow==2.10.0
 - pytools==2020.4.3
 - Mako==1.1.4

2.环境准备-tensorRT加速(optional)

 跑Bert-binary-classification-en-v2.tar代码加速还是比较明显的
 15:43 16:07 -> 24分钟 (未装tensorRT)
 14:31 14:45 -> 14分钟（装tensorRT）

  - pycuda-2020.1下载解压编码
  '''
   python3 configure.py --cuda-root=/usr/local/cuda-11.3
   make -j 4     # 解决各种的关键步骤，先编译
   python3 setup.py install
  '''
  - 然后看https://blog.51cto.com/u_15905131/5918414网址中 二、TensorRT配置即可
  - (遇到libnvinfer.so.7和libnvinfer_plugin.so.7找不到问题，把TensorRT-8.4.0.6/lib/下对应的文件.8改成.7即可)


3.代码准备

 - 把代码和数据放到所有服务器上，确保代码和数据一模一样
 - bert.sh中确保主节点--node_rank=0，从节点--node_rank=1（如果有更多台机器，则=2,3,4等）
 - cd到Bert-binary-classification-en。主节点先sh bert.sh然后从节点sh bert.sh

大功告成！


注：
Bert-binary-classification-en-v1.tar 没有模型保存和加载模块

Bert-binary-classification-en-v2.tar 模型有保存和加载(predict.py)模块【推荐】


