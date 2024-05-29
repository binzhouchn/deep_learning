# Pinecone

```python
import os
import openai
import torch
import numpy as np
import pandas as pd
from meutils.pipe import * 
import json
from tqdm import tqdm
import time
postdata.head()
'''
    id	 values	                                            metadata
0	0	[0.007221237290650606, -0.02768983691930771,...	    {'text': 'xx1'}
1	1	[-0.012209651991724968, -0.011560298502445221,...	{'text': 'xx2'}
2	2	[0.04613766074180603, -0.027380110695958138, ...	{'text': 'xx3'}
3	3	[0.014231022447347641, -0.017830614000558853, ...	{'text': 'xx4'}
4	4	[-0.007622462697327137, -0.013061492703855038,...	{'text': '三、信息披露及备置地点'}
'''
q = '兴化股份2020年营业利润率是多少?保留2位小数。'
```

```python
import pinecone 
pinecone.init(api_key='fb3xxx', environment='gcp-starter')  #api_key和environment可以从https://app.pinecone.io/organizations/-NbIxSm2UEI-1xS_DH7O/projects/gcp-starter:8f2dc48/keys看到
#1.创建和删除index
##1.1创建
index_name = 'first-pinecone-index'
# only create index if it doesn't exist
if index_name not in pinecone.list_indexes():
    pinecone.create_index(
        name=index_name,
        dimension=len(postdata.iloc[0]['values']),
        metric='cosine'
    )
    # wait a moment for the index to be fully initialized
    time.sleep(1)

##1.2删除
#pinecone.delete_index(index_name)

#2.DescribeIndexStats
index = pinecone.Index(index_name) 
index.describe_index_stats()

#3.插入数据
index = pinecone.Index(index_name) 
'''
插入数据示例：
vectors=[
        {
        'id':'vec1', 
        'values':[0.1, 0.2, 0.3, 0.4], 
        'metadata':{'genre': 'drama'},
           'sparse_values':
           {'indices': [10, 45, 16],
           'values':  [0.5, 0.5, 0.2]}},
        {'id':'vec2', 
        'values':[0.2, 0.3, 0.4, 0.5], 
        'metadata':{'genre': 'action'},
           'sparse_values':
           {'indices': [15, 40, 11],
           'values':  [0.4, 0.5, 0.2]}}
    ]
'''
upsert_response = index.upsert(
    vectors=postdata.to_dict(orient='records')
)

#4.fetch数据by id
#fetch_response = index.fetch(ids=['0', '1'])
#5.删除数据by id
# delete_response = index.delete(ids=['0', '1'])
#6.更新数据by id
# update_response = index.update(
#     id='2',
#     values=[0.1]*768,
#     set_metadata={'text': 'new text blabla'}
# )

#7.查询
xq = model.encode(q, normalize_embeddings=True).tolist()
xc = index.query(xq, top_k=5, include_metadata=True)
xc
'''
{'matches': [{'id': '145',
              'metadata': {'text': '["[纳税主体名称 所得税税率]" "[兴化股份 25%]" "[兴化化工 '
                                   '15%]"]'},
              'score': 0.811692715,
              'values': []},
             {'id': '110',
              'metadata': {'text': '3、合并利润表单位：元'},
              'score': 0.798357189,
              'values': []},
             {'id': '1',
              'metadata': {'text': '["[股票简称 兴化股份 股票代码 002109]" "[股票上市证券交易所 '
                                   '深圳证券交易所  ]" "[公司的中文名称 陕西兴化化学股份有限公司  ]" '
                                   '"[公司的中文简称 兴化股份  ]" "[公司的外文名称（如有） SHAANXI '
                                   'XINGHUA CHEMISTRY CO.LTD  ]" '
                                   '"[公司的外文名称缩写（如有） XINGHUA CHEMISTRY  ]" '
                                   '"[公司的法定代表人 樊洺僖  ]" "[注册地址 陕西省咸阳市兴平市东城区  ]" '
                                   '"[注册地址的邮政编码 713100  ]" "[办公地址 '
                                   '陕西省咸阳市兴平市东城区迎宾大道  ]" "[办公地址的邮政编码 713100  '
                                   ']" "[公司网址 www.snxhchem.com  ]" "[电子信箱 '
                                   'snxhchem002109@163.com  ]"]'},
              'score': 0.787375271,
              'values': []},
             {'id': '356',
              'metadata': {'text': '3、营业收入和营业成本单位：元'},
              'score': 0.786100328,
              'values': []},
             {'id': '0',
              'metadata': {'text': '第二节公司简介和主要财务指标一、公司信息'},
              'score': 0.78118217,
              'values': []}],
 'namespace': ''}
'''
```


### 具体可看/Users/zhoubin/PycharmProjects/temp/ChatLLM目录下生成向量并使用向量数据库pinecone.ipynb文件