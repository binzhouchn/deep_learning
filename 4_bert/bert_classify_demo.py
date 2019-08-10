# coding: utf-8
# File: bert_classify_demo.py
# Author: zhoubin
# Date: 20190810

from sklearn.model_selection import train_test_split
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
from datetime import datetime

import bert
from bert import run_classifier
from bert import optimization
from bert import tokenization
from bert import run_classifier_with_tfhub

#########################1. 数据读取#########################

# train, test数据如下，以下展示的是train.sample.head()
'''
                           sentence	                    	  label
19977	"Proximity" tells of a convict (Lowe) who thin...       0
17485	If you can believe it, *another* group of teen...	    0
20116	Well, what can I say having just watched this ...	    1
7347	Strictly a routine, by-the-numbers western (di...	    0
5362	In his 1966 film "Blow Up", Antonioni had his ...       0
'''

#########################2. Data Preprocessing#########################
# Use the InputExample class from BERT's run_classifier code to create examples from the data
train_InputExamples = train.apply(lambda x: bert.run_classifier.InputExample(guid=None, # Globally unique ID for bookkeeping, unused in this example
                                                                   text_a = x['sentence'],
                                                                   text_b = None,
                                                                   label = x['label']), axis = 1)

test_InputExamples = test.apply(lambda x: bert.run_classifier.InputExample(guid=None,
                                                                   text_a = x['sentence'],
                                                                   text_b = None,
                                                                   label = x['label']), axis = 1)

# 以上代码也可以自己写一个InputExample的类，以下是InputExample源码
class InputExample(object):
  """A single training/test example for simple sequence classification."""

  def __init__(self, guid, text_a, text_b=None, label=None):
    """Constructs a InputExample.

    Args:
      guid: Unique id for the example.
      text_a: string. The untokenized text of the first sequence. For single
        sequence tasks, only this sequence must be specified.
      text_b: (Optional) string. The untokenized text of the second sequence.
        Only must be specified for sequence pair tasks.
      label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """
    self.guid = guid
    self.text_a = text_a
    self.text_b = text_b
    self.label = label
train_InputExamples = train.apply(lambda x: InputExample(guid=None, # Globally unique ID for bookkeeping, unused in this example
                                                           text_a = x['sentence'],
                                                           text_b = None,
                                                           label = x['label']), axis = 1)

test_InputExamples = test.apply(lambda x: InputExample(guid=None,
                                                           text_a = x['sentence'],
                                                           text_b = None,
                                                           label = x['label']), axis = 1)

# 有两种方法可以下载和使用预先训练的BERT模型：
'''
1.直接使用tensorflow-hub:
以下预训练模型可供选择。
    BERT-Base, Uncased：12层，768隐藏，12头，110M参数
    BERT-Large, Uncased：24层，1024个隐藏，16个头，340M参数
    BERT-Base, Cased：12层，768隐藏，12头，110M参数
    BERT-Large, Cased：24层，1024个隐藏，16个头，340M参数
    BERT-Base, Multilingual Case：104种语言，12层，768隐藏，12头，110M参数
    BERT-Base, Chinese：中文简体和繁体，12层，768隐藏，12头，110M参数
将使用基本型号：'uncased_L-12_H-768_A-12'
'''
BERT_MODEL_HUB = 'https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1'
tokenizer = run_classifier_with_tfhub.create_tokenizer_from_hub_module(BERT_MODEL_HUB)
'''
2.手动下载BERT模型文件：下载并保存到目录中并解压缩:
当下载BERT-Base, Uncased,这些有3个重要文件
'''
BERT_VOCAB= 'uncased_L-12_H-768_A-12/vocab.txt'
BERT_INIT_CHKPNT = 'uncased_L-12_H-768_A-12/bert_model.ckpt'
BERT_CONFIG = 'uncased_L-12_H-768_A-12/bert_config.json'
tokenization.validate_case_matches_checkpoint(True, BERT_INIT_CHKPNT)
tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file_path, do_lower_case=True)

'''
使用tokenizer，将convert_examples_to_features 在示例上调用方法将它们转换为BERT理解的功能。
此方法添加了BERT用于识别句子开始和结束的特殊“CLS”和“SEP”标记。它还为每个输入添加“index”和“segment”标记。
因此根据BERT格式化输入的所有工作都由此函数完成。
'''

