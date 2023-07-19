#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3

export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL
export TORCH_DISTRIBUTED_DEBUG=INFO
#export NCCL_IB_DISABLE=1
#export NCCL_NET=IB
#export NCCL_SOCKET_IFNAME=enp2s0f0
#export NCCL_SOCKET_IFNAME="^lo,docker,virbr,vmnet,vboxnet,wl,ww,ppp"
export NCCL_SOCKET_IFNAME=en,eth,em,bond,ens

#单机多卡
#deepspeed bert_ds.py --deepspeed --deepspeed_config ds_config.json
#多机多卡
deepspeed --hostfile hostfile --master_addr xx.xxx.72.6 --master_port=29346 --include="xx.xxx.72.6:0,1,2,3@xx.xxx.72.5:0,1,2,3" bert_ds.py --deepspeed --deepspeed_config ds_config.json