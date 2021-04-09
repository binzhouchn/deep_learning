#!/usr/bin/env python
# coding: utf-8

# ## 载入依赖

# ```
# 镜像：binzhouchn/dl:torch1.7.1-tf2.3.1-xgb-lgb-cuda10.1-cudnn7
# ```

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
pd.set_option('display.max_columns', 500)
import cufflinks
import cufflinks as cf
import plotly.figure_factory as ff
import os
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from sklearn import preprocessing
import xgboost as xgb
from sklearn.metrics import *
print("XGBoost version:", xgb.__version__)
from tqdm.notebook import tqdm
import tensorflow as tf
import cudf
import datatable as dtable
import treelite#1.0.0
import treelite_runtime#1.0.0

# ## 设置GPU

# In[2]:


gpus = tf.config.experimental.list_physical_devices('GPU')
print('all gpus: ', gpus)
tf.config.experimental.set_visible_devices(gpus[-2], 'GPU') #GPU='2'
print(tf.config.get_visible_devices('GPU'))


# ## 1. 读取数据

# In[3]:


path = 'data/'
# train = cudf.read_csv(path+'train.csv').to_pandas()#用GPU加速读取
train = dtable.fread(path+'train.csv').to_pandas()#用多线程读取
features = pd.read_csv(path+'features.csv')
example_test = pd.read_csv(path+'example_test.csv')
sample_prediction_df = pd.read_csv(path+'example_sample_submission.csv')
print ("Data is loaded!")


# ### 1.1压缩数据

# In[42]:


def reduce_mem_usage(props):
    start_mem_usg = props.memory_usage().sum() / 1024**2 
    print("Memory usage of properties dataframe is :",start_mem_usg," MB")
    NAlist = [] # Keeps track of columns that have missing values filled in. 
    for col in props.columns:
        if props[col].dtype != object:  # Exclude strings
            
            # Print current column type
            # print("******************************")
            # print("Column: ",col)
            # print("dtype before: ",props[col].dtype)
            
            # make variables for Int, max and min
            IsInt = False
            mx = props[col].max()
            mn = props[col].min()
            
            # Integer does not support NA, therefore, NA needs to be filled
            if not np.isfinite(props[col]).all(): 
                NAlist.append(col)
                props[col].fillna(mn-1,inplace=True)  
                   
            # test if column can be converted to an integer
            asint = props[col].fillna(0).astype(np.int64)
            result = (props[col] - asint)
            result = result.sum()
            if result > -0.01 and result < 0.01:
                IsInt = True

            
            # Make Integer/unsigned Integer datatypes
            if IsInt:
                if mn >= 0:
                    if mx < 255:
                        props[col] = props[col].astype(np.uint8)
                    elif mx < 65535:
                        props[col] = props[col].astype(np.uint16)
                    elif mx < 4294967295:
                        props[col] = props[col].astype(np.uint32)
                    else:
                        props[col] = props[col].astype(np.uint64)
                else:
                    if mn > np.iinfo(np.int8).min and mx < np.iinfo(np.int8).max:
                        props[col] = props[col].astype(np.int8)
                    elif mn > np.iinfo(np.int16).min and mx < np.iinfo(np.int16).max:
                        props[col] = props[col].astype(np.int16)
                    elif mn > np.iinfo(np.int32).min and mx < np.iinfo(np.int32).max:
                        props[col] = props[col].astype(np.int32)
                    elif mn > np.iinfo(np.int64).min and mx < np.iinfo(np.int64).max:
                        props[col] = props[col].astype(np.int64)    
            
            # Make float datatypes 32 bit
            else:
                props[col] = props[col].astype(np.float32)
            
            # Print new column type
            # print("dtype after: ",props[col].dtype)
            # print("******************************")
    
    # Print final result
    print("___MEMORY USAGE AFTER COMPLETION:___")
    mem_usg = props.memory_usage().sum() / 1024**2 
    print("Memory usage is: ",mem_usg," MB")
    print("This is ",100*mem_usg/start_mem_usg,"% of the initial size")
    return props, NAlist

train, _ = reduce_mem_usage(train)


# ## 2. 设置训练集与标签
# 注意到这里没有进行特种工程，实际上线下进行的少量特征工程可以带来线下分数的提升，但是由于线下没有提升，所以最后没有加入最终的模型。这里去掉了利用模型判断出来的 outliers(异常数据)，可以带来200分的线上提升，具体算法可以[参考这里](https://www.kaggle.com/snippsy/feature-importance-over-time-for-outlier-detection) 。
# 这里的标签选择了 resp_3，而不是 resp，原因参见文章第一部分的叙述。

# In[55]:


exclude = set([2,5,19,26,29,36,37,43,63,77,87,173,262,264,268,270,276,294,347,499])
train = train[~train.date.isin(exclude)]

features = [c for c in train.columns if 'feature' in c]

f_mean = train[features[1:]].mean()
train[features[1:]] = train[features[1:]].fillna(f_mean)

train = train[train.weight>0]

train['action'] = ((train['resp'].values) > 0).astype('int')
train['action1'] = ((train['resp_1'].values) > 0).astype('int')
train['action2'] = ((train['resp_2'].values) > 0).astype('int')
train['action3'] = ((train['resp_3'].values) > 0).astype('int')
train['action4'] = ((train['resp_4'].values) > 0).astype('int')

X = train.loc[:, train.columns.str.contains('feature')]
y = train.loc[:, 'action3'].astype('int').values


# ## 3. XGBOOST模型与训练
# 这里加入的超参是L1、L2正则化，10是线下测出来最好的参数，线上也带来了最好的分数。

# In[75]:


clf2 = xgb.XGBClassifier(
      n_estimators=400,
      max_depth=11,
      learning_rate=0.05,
      subsample=0.90,
      colsample_bytree=0.7,
      missing=-999,
      random_state=2020,
      tree_method='gpu_hist',  # THE MAGICAL PARAMETER
      reg_alpha=10,
      reg_lambda=10,
)
clf2.fit(X, y)
clf2.save_model('tmp/mymodel.model')

model = treelite.Model.load('tmp/mymodel.model', model_format='xgboost')
model.export_lib(toolchain='gcc', libpath='tmp/mymodel.so',
                 params={'parallel_comp': 32}, verbose=True)
predictor = treelite_runtime.Predictor('tmp/mymodel.so', verbose=True)
# ## 4. 输出结果

# In[116]:


# res = []
# tofill = f_mean.values.reshape((1,-1))
# for i in tqdm(range(1, len(example_test)+1)):
#     test_df = example_test.iloc[i-1:i]
#     if test_df['weight'].values[0] == 0:
#         res.append(0)
#     else:
#         X_test = test_df.loc[:, features].values
#         if np.isnan(X_test.sum()):
#             X_test[0,1:] = np.where(np.isnan(X_test[0,1:]), tofill, X_test[0,1:])
#         batch = treelite_runtime.DMatrix(X_test) #shape(1,130)
#         y_preds = int((predictor.predict(batch))>0.5)
#         res.append(y_preds)
##sum(res)
##output: 6396

# In[ ]:

#调用janestreet API
import janestreet
env = janestreet.make_env() # initialize the environment
iter_test = env.iter_test() # an iterator which loops over the test set

tofill = f_mean.values.reshape((1,-1))
for (test_df, sample_prediction_df) in iter_test:
    
    
    if test_df['weight'].values[0] == 0:
        sample_prediction_df.action = 0
    else:
        X_test = test_df.loc[:, features].values
        if np.isnan(X_test.sum()):
            X_test[0,1:] = np.where(np.isnan(X_test[0,1:]), tofill, X_test[0,1:])
        batch = treelite_runtime.DMatrix(X_test)
        y_preds = int((predictor.predict(batch))>0.5)
        sample_prediction_df.action = y_preds
    env.predict(sample_prediction_df)


# In[ ]:




