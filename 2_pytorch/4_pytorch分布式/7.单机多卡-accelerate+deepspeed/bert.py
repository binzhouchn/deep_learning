#!/usr/bin/env python
# author = 'binzhou'
# time = 2023/7/18
import argparse
import time
import random
import os
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from model import SentimentClassifier
from dataloader import SSTDataset
import deepspeed
from deepspeed.ops.adam import DeepSpeedCPUAdam

from accelerate.state import AcceleratorState


from accelerate import Accelerator, DeepSpeedPlugin
deepspeed_plugin = DeepSpeedPlugin(zero_stage=2, gradient_accumulation_steps=2)
accelerator = Accelerator(mixed_precision='fp16', gradient_accumulation_steps=2, deepspeed_plugin=deepspeed_plugin)


def get_eval_ds_config(offload=None, stage=3):
    deepspeed_states = AcceleratorState().deepspeed_plugin

    device = "cpu" if offload else "none"
    zero_opt_dict = {
        "stage": stage,
        "stage3_param_persistence_threshold": 1e4,
        "offload_param": {
            "device": device,
            "pin_memory": True
        },
        "offload_optimizer":{
            "device": device,
            "pin_memory": True
        }
    }
    return {
        "train_micro_batch_size_per_gpu": deepspeed_states.deepspeed_config['train_micro_batch_size_per_gpu'],
        "steps_per_print": 10,
        "zero_optimization": zero_opt_dict,
        "bf16": {
            "enabled": True
        },
        "gradient_clipping": 1.0,
        "prescale_gradients": False,
        "wall_clock_breakdown": False
    }
# def get_parms(model, lr):
#     no_decay = ["bias", "LayerNorm.weight"]
#     params = [
#     {
#         "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
#         "weight_decay": 0.0,
#         "lr": lr
#     },
#     {
#         "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
#         "weight_decay": 0.0,
#         "lr": lr
#     },
#     ]
#     return params

# def build_optimizer():
#     params = get_parms(ddp_net, 5e-6)
#     deepspeed_states = AcceleratorState().deepspeed_plugin
#     if deepspeed_states.deepspeed_config['zero_optimization']['offload_optimizer']['device'] in ('none', None):
#         return torch.optim.AdamW(params)
#     return DeepSpeedCPUAdam(params)

def synchronize_if_distributed():
    if accelerator.use_distributed:
        accelerator.wait_for_everyone()

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


ckpt_path = 'model.pth'
best_acc = 0

if __name__ == '__main__':
    set_seed(42)
    
    device_id = accelerator.device

    # 单机单卡跑模型
    ddp_net = SentimentClassifier().to(device_id)
    batch_size = 16
    criterion = nn.CrossEntropyLoss()

    optim = torch.optim.Adam(ddp_net.parameters(), lr=5e-6)

    # setup deepspeed
    deepspeed_states = AcceleratorState().deepspeed_plugin
    deepspeed_states.deepspeed_config['train_micro_batch_size_per_gpu'] = batch_size
    deepspeed_states.deepspeed_config['checkpoint'] = {'use_node_local_storage': True}


    train_set = SSTDataset(filename='data/SST-2/train.tsv')
    train_loader = DataLoader(train_set, batch_size=batch_size)  # 注意这里的batch_size是每个GPU上的batch_size
    val_set = SSTDataset(filename='data/SST-2/dev.tsv')
    val_loader = DataLoader(val_set, batch_size=batch_size)

    #new add
    ddp_net, optim, train_loader, val_loader = accelerator.prepare(ddp_net, optim, train_loader, val_loader)

    # get untrainable model
    # eval_ds_config = get_eval_ds_config(offload=True)
    # ddp_net, *_ = deepspeed.initialize(model=ddp_net, config=eval_ds_config)

    for epoch in range(5):
        t = time.time()
        ddp_net.train()
        for index, (tokens, labels) in enumerate(train_loader):
            optim.zero_grad()
            labels = labels.to(device_id)
            input_ids = tokens['input_ids'].squeeze(1).to(device_id)
            attention_mask = tokens['attention_mask'].squeeze(1).to(device_id)
            token_type_ids = tokens['token_type_ids'].squeeze(1).to(device_id)
            logits = ddp_net(input_ids, attention_mask, token_type_ids)  # tokens -> [B,max_len]
            loss = criterion(logits, labels)
            #loss.backward()
            accelerator.backward(loss)
            optim.step()

            if (index+1) % 100 == 0:
                pred_labels = torch.argmax(logits, dim=1)  # 预测出的label
                pred_labels, labels = accelerator.gather_for_metrics((pred_labels, labels))
                acc = torch.sum(pred_labels == labels) / len(pred_labels)  # acc
                print(f"{accelerator.device} Train... [epoch {epoch + 1}/{epoch_num}, step {i + 1}/{len(train_loader)}]\t[loss {loss.item()} train_acc {acc} ]")
        print("耗时:{}".format(time.time() - t))

        ddp_net.eval()
        with torch.no_grad():
            total_acc = 0
            count = 0
            for i, (tokens, labels) in enumerate(val_loader):
                labels = labels.to(device_id)
                input_ids = tokens['input_ids'].squeeze(1).to(device_id)
                attention_mask = tokens['attention_mask'].squeeze(1).to(device_id)
                token_type_ids = tokens['token_type_ids'].squeeze(1).to(device_id)
                logits = ddp_net(input_ids, attention_mask, token_type_ids)  # tokens -> [B,max_len]
                # count += len(input_ids)
                pred_labels = torch.argmax(logits, dim=1)  # 预测出的label

                # 合并每个GPU的验证数据
                pred_labels, labels = accelerator.gather_for_metrics((pred_labels, labels))
                count += labels.size(0)
                total_acc += torch.sum(pred_labels == labels)
            
            mean_acc = total_acc.item() / count
            accelerator.print(f'epoch {epoch + 1}/{5}, acc = {mean_acc:.2f}')
            # 等待每个GPU上的模型执行完当前的epoch，并进行合并同步
            # accelerator.wait_for_everyone() 
            synchronize_if_distributed()
            ddp_net = accelerator.unwrap_model(ddp_net)
            if mean_acc > best_acc:
                best_acc = mean_acc
                accelerator.save(ddp_net.state_dict(), ckpt_path) 
            '''
            通过accelerate加载模型
            net = accelerator.unwrap_model(ddp_net)
            net.load_state_dict(torch.load(ckpt_path))
            '''

'''
#运行该bert.py文件
accelerate launch bert.py
'''
