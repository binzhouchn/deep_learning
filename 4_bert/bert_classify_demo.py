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

