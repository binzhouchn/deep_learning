# 环境

python3.9.12(anaconda版本2022.05)

torch==1.12.1+cu102
torchvision==0.13.1+cu102

# 启动（以单机两个GPU为例）

```
export CUDA_VISIBLE_DEVICES=2,3
torchrun --nproc_per_node=2 main.py
```

10000数据 单卡一个GPU 23.4s
10000数据 单卡两个GPU 14.1s
10000数据 单卡三个GPU 11.5s
10000数据 单卡四个GPU 10.3s





-----------------------------------------------------------------------------------------
# 参考网址
[PyTorch 并行训练极简 Demo](https://mp.weixin.qq.com/s?__biz=MzI5MDUyMDIxNA==&mid=2247638753&idx=3&sn=59705f871ebd57f3f85040527bdeeb0b&chksm=ec123318db65ba0ef2940499898f0ad0d1d3bf34167d89b880fabc22215512e2be0f35fe879a&mpshare=1&scene=23&srcid=0314wmvKMWBN8TQzyMPt3BJE&sharer_sharetime=1679562302804&sharer_shareid=d3bb90e2f60a69fc71407a2191356565#rd)