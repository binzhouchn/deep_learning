1.配置accelerate_config.yaml

如果是有两台机器，每台机器4张V100卡，则
1.1 main_process_ip写主机器的ip地址
1.2 num_machines: 2
1.3 num_processes: 8
1.4
主节点的machine_rank: 0
从节点的machine_rank: 1

其他都一样。

2.启动，

先主节点sh bert_accelerate.sh
然后再从节点sh bert_accelerate.sh

3.提升情况

相比于之前装tensorRT还能进一步提升
16:19 16:21 -> 2分钟(accelerate)






备注：其他模型数据工具等代码见Bert-binary-classification-en-v2.tar