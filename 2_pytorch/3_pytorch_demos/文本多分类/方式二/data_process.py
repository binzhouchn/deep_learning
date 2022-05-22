#encoding:utf-8
from tqdm import tqdm
from pybert.common.tools import save_pickle
from pybert.configs.basic_config import config
from pybert.model.bert_for_multi_label import BertForMultiLable
from pybert.preprocessing.preprocessor import EnglishPreProcessor

from pybert.io.task_data import TaskData
data = TaskData()
#train
targets_train, sentences_train = data.read_data_THUCNews(raw_data_path='pybert/dataset/train.csv',
                                    preprocessor=EnglishPreProcessor(),
                                    is_train=True)
#dev
targets_dev, sentences_dev = data.read_data_THUCNews(raw_data_path='pybert/dataset/dev.csv',
                                    preprocessor=EnglishPreProcessor(),
                                    is_train=True)

train = []
dev = []
data_dir=config['data_dir']
data_name='THUCNews'

for step,(data_x, data_y) in tqdm(enumerate(zip(sentences_train, targets_train))):
    train.append((data_x, data_y))
for step,(data_x, data_y) in tqdm(enumerate(zip(sentences_dev, targets_dev))):
    dev.append((data_x, data_y))

train_path = data_dir / f"{data_name}.train.pkl"
valid_path = data_dir / f"{data_name}.valid.pkl"
save_pickle(data=train, file_path=train_path)
save_pickle(data=dev, file_path=valid_path)