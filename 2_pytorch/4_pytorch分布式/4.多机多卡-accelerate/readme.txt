1.配置accelerate_config.yaml

主节点的machine_rank: 0
从节点的machine_rank: 1
其他都一样。

2.启动，

先主节点sh bert_accelerate.sh
然后再从节点sh bert_accelerate.sh

3.提升情况

相比于之前装tensorRT还能进一步提升
17:38 17:43 -> 5分钟(accelerate)






备注：其他模型数据工具等代码见Bert-binary-classification-en-v2.tar