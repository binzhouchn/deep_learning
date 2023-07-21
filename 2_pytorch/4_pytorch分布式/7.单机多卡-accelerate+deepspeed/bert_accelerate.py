#!/usr/bin/env python
# author = 'binzhou'
# time = 2023/7/20
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
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate.state import AcceleratorState

from accelerate.utils import DummyOptim


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
    batch_size = 16
    # deepspeed_plugin = DeepSpeedPlugin(zero_stage=2, gradient_accumulation_steps=2)
    # accelerator = Accelerator(mixed_precision='fp16', deepspeed_plugin=deepspeed_plugin)
    accelerator = Accelerator()

    device_id = accelerator.device
    # 跑模型
    ddp_net = SentimentClassifier()#.to(device_id)
    criterion = nn.CrossEntropyLoss()
    # optim = torch.optim.Adam(ddp_net.parameters(), lr=5e-6)

    #new add
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in ddp_net.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
        {
            "params": [p for n, p in ddp_net.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer_cls = (
        torch.optim.AdamW
        if accelerator.state.deepspeed_plugin is None
        or "optimizer" not in accelerator.state.deepspeed_plugin.deepspeed_config
        else DummyOptim
    )
    optim = optimizer_cls(optimizer_grouped_parameters, lr=5e-6)

    train_set = SSTDataset(filename='data/SST-2/train.tsv')
    train_loader = DataLoader(train_set, batch_size=batch_size)  # 注意这里的batch_size是每个GPU上的batch_size
    val_set = SSTDataset(filename='data/SST-2/dev.tsv')
    val_loader = DataLoader(val_set, batch_size=batch_size)

    #new add
    model_engine, optim, train_loader, val_loader = accelerator.prepare(ddp_net, optim, train_loader, val_loader)

    synchronize_if_distributed()#new add

    total_epoch = 5
    for epoch in range(total_epoch):
        t = time.time()
        model_engine.train()
        for index, (tokens, labels) in enumerate(train_loader):
            optim.zero_grad()
            labels = labels.to(device_id)
            input_ids = tokens['input_ids'].squeeze(1).to(device_id)
            attention_mask = tokens['attention_mask'].squeeze(1).to(device_id)
            token_type_ids = tokens['token_type_ids'].squeeze(1).to(device_id)
            logits = model_engine(input_ids, attention_mask, token_type_ids)  # tokens -> [B,max_len]
            loss = criterion(logits, labels)
            accelerator.backward(loss)
            optim.step()

            if (index+1) % 100 == 0:
                pred_labels = torch.argmax(logits, dim=1)  # 预测出的label
                pred_labels, labels = accelerator.gather_for_metrics((pred_labels, labels))
                acc = torch.sum(pred_labels == labels) / len(pred_labels)  # acc
                print(f"{accelerator.device} Train... [epoch {epoch + 1}/{total_epoch}, step {index + 1}/{len(train_loader)}]\t[loss {loss.item()} train_acc {acc} ]")
        print("耗时:{}".format(time.time() - t))

        model_engine.eval()
        with torch.no_grad():
            total_acc = 0
            count = 0
            for i, (tokens, labels) in enumerate(val_loader):
                labels = labels.to(device_id)
                input_ids = tokens['input_ids'].squeeze(1).to(device_id)
                attention_mask = tokens['attention_mask'].squeeze(1).to(device_id)
                token_type_ids = tokens['token_type_ids'].squeeze(1).to(device_id)
                logits = model_engine(input_ids, attention_mask, token_type_ids)  # tokens -> [B,max_len]
                # count += len(input_ids)
                pred_labels = torch.argmax(logits, dim=1)  # 预测出的label

                # 合并每个GPU的验证数据
                pred_labels, labels = accelerator.gather_for_metrics((pred_labels, labels))
                count += labels.size(0)
                total_acc += torch.sum(pred_labels == labels)
            
            mean_acc = total_acc.item() / count
            accelerator.print(f'epoch {epoch + 1}/{total_epoch}, acc = {mean_acc:.2f}')
            # 等待每个GPU上的模型执行完当前的epoch，并进行合并同步
            # accelerator.wait_for_everyone() 
            synchronize_if_distributed()
            # ddp_net = accelerator.unwrap_model(ddp_net)
            if mean_acc > best_acc:
                best_acc = mean_acc
                print(f'epoch {epoch + 1}/{total_epoch}, acc = {mean_acc:.2f}')
                print('save here!')
                # accelerator.save(ddp_net.state_dict(), ckpt_path) 
    synchronize_if_distributed()

