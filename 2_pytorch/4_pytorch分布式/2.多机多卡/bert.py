#!/usr/bin/env python
# author = 'binzhou'
# time = 2023/6/18
import argparse
import time
import random
import os
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
from model import SentimentClassifier
from dataloader import SSTDataset
from torch.utils.data.distributed import DistributedSampler  # 负责分布式dataloader创建，也就是实现上面提到的partition。

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def init_distributed_mode(args):
    if not torch.distributed.is_available():
        raise ValueError('This machine is not supported the DDP!')

    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.local_rank = int(os.environ['LOCAL_RANK'])
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    print(args)
    args.distributed = True
    args.dist_url = 'env://'
    args.dist_backend = 'nccl'
    print(f'distributed init (rank {args.rank}): {args.dist_url}',
          flush=True)
    dist.init_process_group(backend=args.dist_backend,
    init_method='env://')
    dist.barrier()

def cleanup():
    dist.destroy_process_group()


ckpt_path = 'model.pth'
best_acc = 0

if __name__ == '__main__':
    set_seed(42)
    # 负责创建 args.local_rank 变量，并接受 torch.distributed.launch 注入的值
    parser = argparse.ArgumentParser()
    # 不要改该参数，系统会自动分配
    parser.add_argument('--device', default='cuda',
                        help='device id (i.e. 0 or 0,1 or cpu)')
    # 开启的进程数(注意不是线程),不用设置该参数，会根据nproc_per_node自动设置
    parser.add_argument('--local_rank', type=int, default=0)

    parser.add_argument('--lr', type=float, default=5e-6)
    args = parser.parse_args()
    init_distributed_mode(args)
    device = torch.device(args.device)
    torch.cuda.set_device(args.local_rank)
    args.rank = dist.get_rank()
    print(f"[init] == local rank: {args.local_rank}, global rank: {args.rank} ==")

    # 跑模型
    net = SentimentClassifier().to(device)
    batch_size = 16
    criterion = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(net.parameters(), lr=args.lr)

    # 只master进程做logging，否则输出会很乱
    if args.local_rank == 0:
        tb_writer = SummaryWriter(comment='ddp-training')

    train_set = SSTDataset(filename='data/SST-2/train.tsv')
    # 分布式数据集
    train_sampler = DistributedSampler(train_set)
    train_loader = DataLoader(train_set, sampler=train_sampler, batch_size=batch_size)  # 注意这里的batch_size是每个GPU上的batch_size

    val_set = SSTDataset(filename='data/SST-2/dev.tsv')
    val_sampler = DistributedSampler(val_set)
    val_loader = DataLoader(val_set, sampler=val_sampler, batch_size=batch_size)

    for epoch in range(5):
        train_sampler.set_epoch(epoch)
        #val_sampler.set_epoch(epoch)
        t = time.time()
        for index, (tokens, labels) in enumerate(train_loader):

            optim.zero_grad()
            labels = labels.to(device)
            input_ids = tokens['input_ids'].squeeze(1).to(device)
            attention_mask = tokens['attention_mask'].squeeze(1).to(device)
            token_type_ids = tokens['token_type_ids'].squeeze(1).to(device)
            logits = net(input_ids, attention_mask, token_type_ids)  # tokens -> [B,max_len]
            loss = criterion(logits, labels)
            loss.backward()
            optim.step()

            if index % 100 == 0:
                pred_labels = torch.argmax(logits, dim=1)  # 预测出的label
                acc = torch.sum(pred_labels == labels) / len(pred_labels)  # acc
                #写log
                if args.local_rank == 0:
                    tb_writer.add_scalar('acc', acc.item(), index)
                print('gpu:{} epoch:{} pred_acc:{}'.format(args.rank, epoch, acc))
        print("耗时:{}".format(time.time() - t))
        with torch.no_grad():
            mean_acc = 0
            count = 0
            for i, (tokens, labels) in enumerate(val_loader):
                labels = labels.to(device)
                input_ids = tokens['input_ids'].squeeze(1).to(device)
                attention_mask = tokens['attention_mask'].squeeze(1).to(device)
                token_type_ids = tokens['token_type_ids'].squeeze(1).to(device)
                logits = net(input_ids, attention_mask, token_type_ids)  # tokens -> [B,max_len]
                count += len(input_ids)
                pred_labels = torch.argmax(logits, dim=1)  # 预测出的label
                mean_acc += torch.sum(pred_labels == labels)
            valid_acc = mean_acc / count
            print("valid acc: ", mean_acc / count)

            #保存模型
            if args.rank == 0 and valid_acc > best_acc:
                best_acc = valid_acc
                print('best_acc: ', best_acc)
                torch.save(net.state_dict(), ckpt_path)#net.module.cpu().state_dict()

            dist.barrier()

            device_id = args.rank % torch.cuda.device_count()
            map_location = {'cuda:0': f'cuda:{device_id}'}
            state_dict = torch.load(ckpt_path, map_location=map_location)
            print(f'rank {args.rank}: {state_dict}')
            net.load_state_dict(state_dict)

    if args.local_rank == 0:
        tb_writer.close()
    cleanup()