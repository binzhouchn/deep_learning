#encoding:utf-8
import torch
import time
import warnings
from pathlib import Path
from argparse import ArgumentParser
from pybert.train.losses import BCEWithLogLoss,CrossEntropy
from pybert.train.trainer import Trainer
from torch.utils.data import DataLoader
from pybert.io.utils import collate_fn
from pybert.io.bert_processor import BertProcessor
from pybert.common.tools import init_logger, logger
from pybert.common.tools import seed_everything
from pybert.configs.basic_config import config
from pybert.model.bert_for_multi_label import BertForMultiLable
from pybert.preprocessing.preprocessor import EnglishPreProcessor
from pybert.callback.modelcheckpoint import ModelCheckpoint
from pybert.callback.trainingmonitor import TrainingMonitor
from pybert.train.metrics import AUC, AccuracyThresh, MultiLabelReport,AccuracyMultilabel,F1Score,ClassReport
from pybert.callback.optimizater.adamw import AdamW
from pybert.callback.lr_schedulers import get_linear_schedule_with_warmup
from torch.utils.data import RandomSampler, SequentialSampler

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0" #家里就一块1080ti GPU

warnings.filterwarnings("ignore")

class args:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    arch='bert'
    save_best=True
    do_lower_case=True
    data_name='THUCNews'
    mode='min'
    monitor='valid_loss'
    epochs=30
    resume_path=''
    predict_checkpoints=0
    local_rank=-1
    _sorted=0
    n_gpu='0'
    gradient_accumulation_steps=1
    train_batch_size=128
    eval_batch_size=128
    train_max_seq_len=40
    eval_max_seq_len=40
    loss_scale=0
    warmup_proportion=0.1
    weight_decay=0.01
    adam_epsilon=1e-8
    grad_clip=1
    learning_rate=2e-5
    seed=42
    fp16=False
    fp16_opt_level='01'

processor = BertProcessor(vocab_path=config['bert_vocab_path'], do_lower_case=args.do_lower_case)
label_list = processor.get_labels()
label2id = {label: i for i, label in enumerate(label_list)}
id2label = {i: label for i, label in enumerate(label_list)}

# 准备train数据集并封装成dataloader
train_data = processor.get_train(config['data_dir'] / f"{args.data_name}.train.pkl")
train_examples = processor.create_examples(lines=train_data,
                                               example_type='train',
                                               cached_examples_file=config[
                                                    'data_dir'] / f"cached_train_examples_{args.arch}")

#第一次需要跑
train_features = processor.create_features(examples=train_examples,
                                               max_seq_len=args.train_max_seq_len,
                                               cached_features_file=config[
                                                    'data_dir'] / "cached_train_features_{}_{}".format(
                                                   args.train_max_seq_len, args.arch
                                               ))
#如果已经跑过一次则直接读取cache文件
# train_features = processor.create_features(examples='',
#                                                max_seq_len=args.train_max_seq_len,
#                                                cached_features_file=config[
#                                                     'data_dir'] / "cached_train_features_{}_{}".format(
#                                                    args.train_max_seq_len, args.arch
#                                                ))
train_dataset = processor.create_dataset(train_features, is_sorted=args._sorted)
train_sampler = RandomSampler(train_dataset)
train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size,
                                  collate_fn=collate_fn)
# 准备dev数据集并封装成dataloader
#第一次跑
valid_data = processor.get_dev(config['data_dir'] / f"{args.data_name}.valid.pkl")
valid_examples = processor.create_examples(lines=valid_data,
                                           example_type='valid',
                                           cached_examples_file=config[
                                            'data_dir'] / f"cached_valid_examples_{args.arch}")

valid_features = processor.create_features(examples=valid_examples,
                                           max_seq_len=args.eval_max_seq_len,
                                           cached_features_file=config[
                                            'data_dir'] / "cached_valid_features_{}_{}".format(
                                               args.eval_max_seq_len, args.arch
                                           ))
#如果已经跑过一次则直接读取cache文件
# valid_features = processor.create_features(examples='',
#                                            max_seq_len=args.eval_max_seq_len,
#                                            cached_features_file=config[
#                                             'data_dir'] / "cached_valid_features_{}_{}".format(
#                                                args.eval_max_seq_len, args.arch
#                                            ))

valid_dataset = processor.create_dataset(valid_features)
valid_sampler = SequentialSampler(valid_dataset)
valid_dataloader = DataLoader(valid_dataset, sampler=valid_sampler, batch_size=args.eval_batch_size,
                              collate_fn=collate_fn)

model = BertForMultiLable.from_pretrained(config['bert_model_dir'], num_labels=len(label_list))
t_total = int(len(train_dataloader) / args.gradient_accumulation_steps * args.epochs)
param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],'weight_decay': args.weight_decay},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]
warmup_steps = int(t_total * args.warmup_proportion)
optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                            num_training_steps=t_total)
train_monitor = TrainingMonitor(file_dir=config['figure_dir'], arch=args.arch)
model_checkpoint = ModelCheckpoint(checkpoint_dir=config['checkpoint_dir'],mode=args.mode,
                                       monitor=args.monitor,arch=args.arch,
                                       save_best_only=args.save_best)
trainer = Trainer(args= args,model=model,logger=logger,criterion=CrossEntropy(),optimizer=optimizer,
                      scheduler=scheduler,early_stopping=None,training_monitor=train_monitor,
                      model_checkpoint=model_checkpoint,
                      batch_metrics=[AccuracyMultilabel()],
                      epoch_metrics=[F1Score(average='micro', task_type='multiclass'),
                                     ClassReport(target_names=label_list)])
trainer.train(train_data=train_dataloader, valid_data=valid_dataloader)