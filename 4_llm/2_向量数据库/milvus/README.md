# Milvus

## 1. 安装

[Install Milvus Standalone with Docker Compose](https://milvus.io/docs/install_standalone-docker.md)<br>
```bash
#Download milvus-standalone-docker-compose.yml and save it as docker-compose.yml manually, or with the following command.
wget https://github.com/milvus-io/milvus/releases/download/v2.2.12/milvus-standalone-docker-compose.yml -O docker-compose.yml
#然后
docker-compose up -d
# check if the containers are up and running.
sudo docker-compose ps
```

## 2. 使用

[Run Milvus using Python](https://milvus.io/docs/example_code.md)

```python
#pymilvus==2.2.14
import os
import openai
import torch
import numpy as np
import pandas as pd
from meutils.pipe import * 
import json
from tqdm import tqdm
import time
import jieba
jieba.initialize()

# 加载m3e-bas句向量转化模型 dim=768
import sentence_transformers
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('/Users/zhoubin/Downloads/m3e-base')
q = '兴化股份2020年营业利润率是多少?保留2位小数。'
entities
'''
    id	  vector	                                            text
0	0	[0.4707290530204773, 0.06718272715806961, 0.32...	第二节公司简介和主要财务指标一、公司信息
1	1	[0.1439879834651947, 0.1646626591682434, 0.847...	股票上市证券交易所 深圳证券交易所
2	2	[1.0262799263000488, 0.525122344493866, 0.4553...	二、联系人和联系方式
3	3	[0.3345898687839508, 0.03860317915678024, 0.34...	 董事会秘书 证券事务代表
4	4	[0.49837177991867065, 0.2881564795970917, 0.59...	三、信息披露及备置地点
'''

#milvus使用
import numpy as np
from pymilvus import (
    connections,
    utility,
    FieldSchema, CollectionSchema, DataType,
    Collection,
)

connections.connect(alias="default", host="0.0.0.0", port="19530")

has = utility.has_collection("hello_milvus")
print(f"Does collection hello_milvus exist in Milvus: {has}")
dim=768
fields = [
    FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, auto_id=False, max_length=32),
    FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dim),
    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=1000)
]

schema = CollectionSchema(fields, "hello_milvus is the simplest demo to introduce the APIs")
hello_milvus = Collection("hello_milvus", schema, consistency_level="Strong")

# insert
import random
insert_result = hello_milvus.insert(entities.to_dict(orient='records'))
# After final entity is inserted, it is best to call flush to have no growing segments left in memory
hello_milvus.flush()  

# Builds indexes on the entities
index = {
    "index_type": "IVF_FLAT",
    "metric_type": "L2",
    "params": {"nlist": 128},
}
hello_milvus.create_index("vector", index)

# search
hello_milvus.load()
search_params = {
    "metric_type": "L2",
    "params": {"nprobe": 10},
}
vectors_to_search = model.encode(q).tolist()
result = hello_milvus.search([vectors_to_search], "vector", search_params, limit=3, output_fields=["text"])
for hits in result:
    for hit in hits:
        print(f"hit: {hit}, text field: {hit.entity.get('text')}")
'''
hit: id: 145, distance: 126.4853744506836, entity: {'text': '["[纳税主体名称 所得税税率]" "[兴化股份 25%]" "[兴化化工 15%]"]'}, text field: ["[纳税主体名称 所得税税率]" "[兴化股份 25%]" "[兴化化工 15%]"]
hit: id: 146, distance: 134.0875244140625, entity: {'text': '2、税收优惠根据陕发改产业确认函[2012]004号文，兴化股份被确认为陕西省符合国家鼓励类目录企业，符合财税[2011]58号《财政部国家税务总局海关总署关于深入实施西部大开发战略有关税收政策问题的通知》和国家税务总局公告2012年第12号《国家税务总局关于深入实施西部大开发战略有关企业所得税问题的公告》的规定，可以享受西部大开发税收优惠。自2011年1月1日至2020年12月31日止，兴化股份'}, text field: 2、税收优惠根据陕发改产业确认函[2012]004号文，兴化股份被确认为陕西省符合国家鼓励类目录企业，符合财税[2011]58号《财政部国家税务总局海关总署关于深入实施西部大开发战略有关税收政策问题的通知》和国家税务总局公告2012年第12号《国家税务总局关于深入实施西部大开发战略有关企业所得税问题的公告》的规定，可以享受西部大开发税收优惠。自2011年1月1日至2020年12月31日止，兴化股份
hit: id: 11, distance: 134.89227294921875, entity: {'text': '["[ 2020年 2019年 本年比上年增减 2018年]" "[营业收入（元） 1939991157.02 1974453085.88 -1.75% 2052627902.53]" "[归属于上市公司股东的净利润（元） 213897539.94 146148587.43 46.36% 238031212.35]" "[归属于上市公司股东的扣除非经常性损益的净利润（元） 211351386.54'}, text field: ["[ 2020年 2019年 本年比上年增减 2018年]" "[营业收入（元） 1939991157.02 1974453085.88 -1.75% 2052627902.53]" "[归属于上市公司股东的净利润（元） 213897539.94 146148587.43 46.36% 238031212.35]" "[归属于上市公司股东的扣除非经常性损益的净利润（元） 211351386.54
'''

# query
expr = expr = f'id in ["10" , "20"]'
result2 = hello_milvus.query(expr=expr, output_fields=["id", "text"])
result2
'''
[{'id': '10',
  'text': '公司聘请的报告期内履行持续督导职责的保荐机构□适用√不适用公司聘请的报告期内履行持续督导职责的财务顾问□适用√不适用六、主要会计数据和财务指标公司是否需追溯调整或重述以前年度会计数据□是√否'},
 {'id': '20',
  'text': '2、主要境外资产情况□适用√不适用三、核心竞争力分析兴化化工作为公司目前唯一的经营实体，其核心竞争力主要体现在其所具有的地理、综合利用、管理和品牌等优势上，具体为：1、地理优势作为煤化工企业，兴化化工具有接近煤炭产地的优势。兴化化工实施工艺优化后，拓宽了煤种和煤炭产地的可选择性，体现出原材料采购的成本优势。目前，兴化化工以就近的华亭、彬长煤为原料煤，而周边同类企业大多以运距较大的陕北煤为原料煤（因'}]
'''
#删除collection
# utility.drop_collection("hello_milvus")
```


### 具体可看/Users/zhoubin/PycharmProjects/temp/ChatLLM目录下生成向量并使用向量数据库milvus.ipynb文件