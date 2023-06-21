【已跑通！】

1.环境准备
 
 - 两台服务器（确保服务器之间是通的，并且关闭防火墙sudo ufw status查看状态），每台2张V100；*.*.72.6 ubuntu16.04lts（主）, *.*.72.7 ubuntu22.04lts（从）或者*.*.72.5 ubuntu16.04lts（从）
 - 以.6和.5为例，这两台都是ubuntu16.04lts, cuda11.3, cudnn8.4.0, 
 - Anaconda3-2022.05-Linux-x86_64.sh
 - pip install torch==1.12.1+cu102 torchvision==0.13.1+cu102 -f https://download.pytorch.org/whl/torch_stable.html -i https://pypi.tuna.tsinghua.edu.cn/simple some-package
 - transformers==4.23.1; tensorboard==2.13.0

2.代码准备

 - 把代码和数据放到所有服务器上，确保代码和数据一模一样
 - bert.sh中确保主节点--node_rank=0，从节点--node_rank=1（如果有更多台机器，则=2,3,4等）
 - cd到Bert-binary-classification-en。主节点先sh bert.sh然后从节点sh bert.sh

大功告成！


注：
Bert-binary-classification-en-v1.tar 没有模型保存和加载模块

Bert-binary-classification-en-v2.tar 模型有保存和加载(predict.py)模块