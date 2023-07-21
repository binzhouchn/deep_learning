只需要主节点运行：bert_accelerate.sh

备注：
1.主从节点放入相同的代码(bert_accelerate.sh, bert_accelerate.py, accelerate_config.yaml, zero_stage2_config.json, hostfile)
2.注意accelerate_config.yaml在不同节点的machine_rank不同，主节点是0，从节点是1,2,3...


运行时长：5分50秒