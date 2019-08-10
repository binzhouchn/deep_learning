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
