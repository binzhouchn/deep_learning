# pgvector

## 1. 安装

```shell
docker pull ankane/pgvector:v0.4.4
```

## 2. 启动docker

```shell
docker run --name postgres15 -e POSTGRES_PASSWORD=123456 -e POSTGRES_USER=root -p 5432:5432 -d ankane/pgvector:v0.4.4
```

## 3. python pgvector使用

```python
postdata.head()
'''
    id	 values	                                            metadata
0	0	[0.007221237290650606, -0.02768983691930771,...	    {'text': 'xx1'}
1	1	[-0.012209651991724968, -0.011560298502445221,...	{'text': 'xx2'}
2	2	[0.04613766074180603, -0.027380110695958138, ...	{'text': 'xx3'}
3	3	[0.014231022447347641, -0.017830614000558853, ...	{'text': 'xx4'}
4	4	[-0.007622462697327137, -0.013061492703855038,...	{'text': '三、信息披露及备置地点'}
'''

#https://github.com/pgvector/pgvector-python/blob/master/examples/sentence_embeddings.py
from pgvector.sqlalchemy import Vector
from sentence_transformers import SentenceTransformer
from sqlalchemy import create_engine, insert, select, text, Integer, String, Text
from sqlalchemy.orm import declarative_base, mapped_column, Session
#连接
databasename='postgres'
user='root'
password='123456'
host='localhost'
engine=create_engine(f'postgresql+psycopg://{user}:{password}@{host}:{5432}/{databasename}')
with engine.connect() as conn:
    conn.execute(text('CREATE EXTENSION IF NOT EXISTS vector'))
    conn.commit()

Base = declarative_base()
class Document(Base):
    __tablename__ = 'document'

    id = mapped_column(Integer, primary_key=True)
    content = mapped_column(Text)
    embedding = mapped_column(Vector(768))


Base.metadata.drop_all(engine)
Base.metadata.create_all(engine)

sentences = postdata['metadata'].map(lambda x : x['text']).to_list()
embeddings = model.encode(sentences)
documents = [dict(content=sentences[i], embedding=embedding) for i, embedding in enumerate(embeddings)]
session = Session(engine)
session.execute(insert(Document), documents)

#向量查询q
query_emb = model.encode(q)
neighbors = session.scalars(select(Document).order_by(Document.embedding.cosine_distance(query_emb)).limit(5))
for neighbor in neighbors:
    print(neighbor.content) #neighbor.id, neighbor.embedding
```


### 具体可看/Users/zhoubin/PycharmProjects/temp/ChatLLM目录下生成向量并使用向量数据库pgvector.ipynb文件