#图像分类总入口(pytorch版)
__author__ = 'zhoubin'
__ctime__ = '20210209'

import os
import random
import torch
from torch import nn
from torchvision import transforms
from PIL import Image, ImageOps, ImageFilter
import torch.utils.data as data
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from tqdm import tqdm, tqdm_notebook
from .utils import AverageMeter, accuracy, save_checkpoint
from .models.resnet import resnet50
from .models.efficientnet import EfficientNet
from .models.byobnet import gernet_l, repvgg_b2

# 如果有GPU则指定需要用哪块
gpu_list = '0'
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list

nn.ConvTranspose2d


class args:
    batch_size = 32
    no_cuda = False
    arch = 'resnet50'  # efficientnet名字和所用到的模型最好对应起来
    num_classes = 2
    # 三维图像的均值 PIL(Image)加载
    global_mean = [0.793, 0.546, 0.502]
    global_std = [0.178, 0.224, 0.241]
    checkpoint = './checkpoints' #模型参数保存地址


########### *1. 读取数据* ###########
## 1.1 定义图片transform方式，进行一定的数据增强(可以单独写个py文件放)
class Resize(object):
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        ratio = self.size[0] / self.size[1]
        w, h = img.size
        if w / h < ratio:
            t = int(h * ratio)
            w_padding = (t - w) // 2
            img = img.crop((-w_padding, 0, w + w_padding, h))
        else:
            t = int(w * ratio)
            h_padding = (t - h) // 2
            img = img.crop((0, -h_padding, w, h + h_padding))
        img = img.resize(self.size, self.interpolation)
        return img


class RandomRotate(object):
    '''
    随机旋转图片
    '''

    def __init__(self, degree, p=0.5):
        self.degree = degree
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            rotate_degree = random.uniform(-1 * self.degree, self.degree)
            img = img.rotate(rotate_degree, Image.BILINEAR)
        return img


class RandomGaussianBlur(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            # 高斯模糊是高斯低通滤波器 不保留细节   高斯滤波是高斯高通滤波器 保留细节
            img = img.filter(ImageFilter.GaussianBlur(radius=random.random()))
        return img


def get_train_transform(mean, std, size):
    train_transform = transforms.Compose([
        Resize((int(size * (256 / 224)), int(size * (256 / 224)))),
        transforms.RandomCrop(size),
        transforms.RandomRotation(degrees=15),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    return train_transform


def get_test_transform(mean, std, size):
    return transforms.Compose([
        Resize((int(size * (256 / 224)), int(size * (256 / 224)))),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def get_transforms(input_size=224, test_size=224, backbone=None):
    mean, std = args.global_mean, args.global_std
    if backbone is not None and backbone in ['pnasnet5large', 'nasnetamobile']:
        mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
    transformations = {}
    transformations['val_train'] = get_train_transform(mean, std, input_size)
    transformations['val_test'] = get_test_transform(mean, std, test_size)
    transformations['test'] = get_test_transform(mean, std, test_size)
    return transformations


transformations = get_transforms(input_size=456, test_size=456)


## 1.2 重写pytorch Dataset，定义训练测试数据标签处理方式(可以单独写个py文件放)
class Dataset(Dataset):
    def __init__(self, img_folder, img_name_and_label: list, transform=None):
        '''
        img_name_and_label:图片名以及图片标签，是个列表比如[(0blgp.png, 1),(smdg8d.png, 0),...]
        transforms: 进行数据增强
        '''
        # 存有图片名和label(用逗号分隔)的一个txt或者csv文件，如果有head则需要去掉
        self.length = len(img_name_and_label)
        self.transform = transform
        self.env = img_name_and_label
        self.img_folder = img_folder

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        # print("train...,index=",index)
        assert index <= len(self), 'index range error'
        try:
            img_path, label = self.env[index]
        except ValueError as e:
            print(e, '; it must be test set, label will be set all to 0')
            img_path = self.env[index]
            label = 0
        img_path = os.path.join(self.img_folder, img_path)
        try:
            img = Image.open(img_path)
        except:
            return self[index + 1]
        if self.transform is not None:
            try:
                img = self.transform(img)
            except Exception as e:
                return self[index + 1]

        return (img, int(label))


## 1.3 读取数据进行transform，load等操作
df = pd.read_csv('ds/train_label.csv')  # '7o93ld.png',1 【1表示含有福，0表示不含】
from sklearn.model_selection import StratifiedKFold

stratifiedKFolds = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)  # val占1/4
for (trn_idx, val_idx) in stratifiedKFolds.split(df.img_id, df.label):
    pass

train_set = Dataset('ds/train/', img_name_and_label=[(x[0], x[1]) for x in df.loc[trn_idx].to_numpy()],
                    transform=transformations['val_train'])
train_loader = data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
val_set = Dataset('ds/train/', img_name_and_label=[(x[0], x[1]) for x in df.loc[val_idx].to_numpy()],
                  transform=transformations['val_test'])
val_loader = data.DataLoader(val_set, batch_size=args.batch_size, shuffle=False)


########## *2. 跑模型(定义损失衡量优化器等)* ##########
def make_model(args, pretrained_model_path='', revise_last=True):
    print("=> creating model '{}'".format(args.arch))
    # 加载预训练模型
    model = resnet50(progress=True)
    # model = EfficientNet.from_name('efficientnet-b5') #图片大小3*456*456
    # model = repvgg_b2() #后面修改最后一层全连接的时候是model.head.fc而不是model.fc因为fc写在另一个名为head的类下
    #####
    if pretrained_model_path:
        print('loading..')
        model.load_state_dict(torch.load(pretrained_model_path))
        print('pretrained model loaded!')
    # 修改最后一层全连接层
    if revise_last:
        fc_inputs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(fc_inputs, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, args.num_classes)
        )
        model.fc.requires_grad = True  # 不一定要加
    return model


model = make_model(args, 'pretrained/resnet50-19c8e357.pth', True)

learning_rate = 0.001
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
# optimizer = torch.optim.SGD(model.parameters(), learning_rate,
#                                 momentum=0.9,
#                                 weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=5, verbose=False)

if not args.no_cuda:
    try:
        model = model.cuda()
    except Exception:
        model = torch.nn.DataParallel(model, device_ids=list(range(args.num_gpu))).cuda()
    criterion = criterion.cuda()
########### RUN
EPOCHS = 100
best_acc = 0
for epo in range(EPOCHS):
    model.train()
    losses = AverageMeter()
    train_acc = AverageMeter()
    val_losses = AverageMeter()
    val_acc = AverageMeter()
    for (inputs, targets) in tqdm_notebook(train_loader):
        if not args.no_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        # inputs,targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)
        # 梯度参数设为0
        optimizer.zero_grad()
        # forward
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        # compute gradient and do SGD step
        loss.backward()
        optimizer.step()
        # measure accuracy and record loss
        acc = accuracy(outputs.data, targets.data)
        losses.update(loss.item(), inputs.size(0))
        train_acc.update(acc.item(), inputs.size(0))
    # Model Evaluating
    model.eval()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            if not args.no_cuda:
                input, target = input.cuda(), target.cuda()
            # compute output
            output = model(input)
            loss = criterion(output, target)
            acc = accuracy(output.data, target.data)
            val_losses.update(loss.item(), input.size(0))
            val_acc.update(acc.item(), input.size(0))
    print('Epoch {} - train loss: {}, train_acc: {}, val loss: {}, val_acc: {}'.format(str(epo), str(losses.avg),
                                                                                       str(train_acc.avg),
                                                                                       str(val_losses.avg),
                                                                                       str(val_acc.avg)))
    scheduler.step(val_losses.avg)

    # save checkpoints
    if val_acc.avg > best_acc:
        best_acc = max(val_acc.avg, best_acc)
        save_checkpoint({
            'fold': 0,
            'epoch': epo + 1,
            'state_dict': model.state_dict(),
            'train_acc': train_acc.avg,
            'acc': val_acc.avg,
            'best_acc': best_acc,
            'optimizer': optimizer.state_dict(),
        }, True, single=True, checkpoint=args.checkpoint)
        print('model saved!')
