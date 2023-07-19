#!/usr/bin/env python
# author = 'binzhou'
# time = 2023/7/19
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
import argparse

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def add_argument():
    parser = argparse.ArgumentParser(description='SST')
    parser.add_argument('-b',
                        '--batch_size',
                        default=16,
                        type=int,
                        help='mini-batch size (default: 16)')
    parser.add_argument('--local_rank',
                        type=int,
                        default=-1,
                        help='local rank passed from distributed launcher')
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    return args


ckpt_path = 'model.pth'
best_acc = 0

if __name__ == '__main__':
    set_seed(42)
    batch_size = 16
    #加载数据
    train_set = SSTDataset(filename='data/SST-2/train.tsv')
    train_loader = DataLoader(train_set, batch_size=batch_size)  # 注意这里的batch_size是每个GPU上的batch_size
    val_set = SSTDataset(filename='data/SST-2/dev.tsv')
    val_loader = DataLoader(val_set, batch_size=batch_size)

    args = add_argument()
    ddp_net = SentimentClassifier()
    parameters = filter(lambda p: p.requires_grad, ddp_net.parameters())
    model_engine, optimizer, trainloader, _ = deepspeed.initialize(
        args=args, model=ddp_net, model_parameters=parameters, training_data=train_set)

    criterion = nn.CrossEntropyLoss()

    # optim = torch.optim.Adam(ddp_net.parameters(), lr=5e-6)

    total_epoch = 5
    for epoch in range(total_epoch):
        t = time.time()
        model_engine.train()
        for index, (tokens, labels) in enumerate(train_loader):
            model_engine.zero_grad()
            labels = labels.to(model_engine.local_rank)
            input_ids = tokens['input_ids'].squeeze(1).to(model_engine.local_rank)
            attention_mask = tokens['attention_mask'].squeeze(1).to(model_engine.local_rank)
            token_type_ids = tokens['token_type_ids'].squeeze(1).to(model_engine.local_rank)
            logits = model_engine(input_ids, attention_mask, token_type_ids)  # tokens -> [B,max_len]
            loss = criterion(logits, labels)
            model_engine.backward(loss)
            model_engine.step()

            if (index+1) % 100 == 0:
                pred_labels = torch.argmax(logits, dim=1)  # 预测出的label
                acc = torch.sum(pred_labels == labels) / len(pred_labels)  # acc
                print(f"{model_engine.local_rank} Train... [epoch {epoch + 1}/{total_epoch}, step {index + 1}/{len(train_loader)}]\t[loss {loss.item()} train_acc {acc} ]")
        print("耗时:{}".format(time.time() - t))

        model_engine.eval()
        with torch.no_grad():
            total_acc = 0
            count = 0
            for i, (tokens, labels) in enumerate(val_loader):
                labels = labels.to(model_engine.local_rank)
                input_ids = tokens['input_ids'].squeeze(1).to(model_engine.local_rank)
                attention_mask = tokens['attention_mask'].squeeze(1).to(model_engine.local_rank)
                token_type_ids = tokens['token_type_ids'].squeeze(1).to(model_engine.local_rank)
                logits = model_engine(input_ids, attention_mask, token_type_ids)  # tokens -> [B,max_len]
                # count += len(input_ids)
                pred_labels = torch.argmax(logits, dim=1)  # 预测出的label

                count += labels.size(0)
                total_acc += torch.sum(pred_labels == labels)
            
            mean_acc = total_acc.item() / count
            print(f'gpu: {model_engine.local_rank}, epoch {epoch + 1}/{5}, acc = {mean_acc:.2f}')
            
            if mean_acc > best_acc:
                best_acc = mean_acc
                print('save here!', model_engine.local_rank)
                # accelerator.save(ddp_net.state_dict(), ckpt_path) 

