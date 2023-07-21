#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3

export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL
export TORCH_DISTRIBUTED_DEBUG=INFO
#export NCCL_IB_DISABLE=1
#export NCCL_SOCKET_IFNAME=enp2s0f0
#export NCCL_SOCKET_IFNAME="^lo,docker,virbr,vmnet,vboxnet,wl,ww,ppp"
export NCCL_SOCKET_IFNAME=en,eth,em,bond,ens


accelerate launch --config_file accelerate_config.yaml bert_accelerate.py