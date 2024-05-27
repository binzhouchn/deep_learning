# 向量数据库

## 1. Elasticsearch

```shell
#1.docker安装
docker pull docker.elastic.co/elasticsearch/elasticsearch:7.16.2
#2.生成一些文件
#参考网址 https://blog.csdn.net/qq_45000856/article/details/125672591
#3.安装一些中文插件
#https://github.com/medcl/elasticsearch-analysis-ik
#https://github.com/tenlee2012/elasticsearch-analysis-hao
#https://github.com/medcl/elasticsearch-analysis-pinyin
#4.启动docker
docker run --name elasticsearch -p 9200:9200 -p 9300:9300 -d -e "discovery.type=single-node" -e ES_JAVA_OPTS="-Xms4g -Xmx8g" --privileged=true -v /Users/zhoubin/es_docker/data:/usr/share/elasticsearch/data -v /Users/zhoubin/es_docker/plugins:/usr/share/elasticsearch/plugins -v /Users/zhoubin/es_docker/config/elasticsearch.yml:/usr/share/elasticsearch/config/elasticsearch.yml  -v /Users/zhoubin/es_docker/config/stopwords:/usr/share/elasticsearch/config/stopwords -v /Users/zhoubin/es_docker/config/synonyms:/usr/share/elasticsearch/config/synonyms docker.elastic.co/elasticsearch/elasticsearch:7.16.2
```

## 2. Faiss

[faiss的python接口使用](https://www.ngui.cc/zz/1772454.html?action=onClick)<br>
[Faiss入门及应用经验记录](https://zhuanlan.zhihu.com/p/357414033)<br>
[向量数据库-Faiss详解](https://blog.csdn.net/HAXIMOF/article/details/134946519)<br>
[faiss tutorial github](https://github.com/facebookresearch/faiss/tree/main/tutorial/python)<br>

## 3. Milvus

[Install Milvus Standalone with Docker Compose](https://milvus.io/docs/install_standalone-docker.md)<br>
[milvus官方文档](https://milvus.io/docs/example_code.md)<br>

## 4. Pinecone

[官网](https://app.pinecone.io/organizations/-NbIxSm2UEI-1xS_DH7O/projects/gcp-starter:8f2dc48/indexes)<br>
[api_key和environment](https://app.pinecone.io/organizations/-NbIxSm2UEI-1xS_DH7O/projects/gcp-starter:8f2dc48/keys)<br>
[Pinecone使用文档](https://docs.pinecone.io/reference/query)<br>

## 5. pgvector

[pgvector github网址](https://github.com/pgvector/pgvector#docker)<br>
[pgvector-python github网址](https://github.com/pgvector/pgvector-python/blob/master/examples/sentence_embeddings.py)<br>

## 6. 云服务zilliz(背后是milvus数据库)

(官网)[https://cloud.zilliz.com.cn/orgs/org-qgyozwourrntbtomshnoof/projects/MA==/clusters]<br>
(zilliz使用文档)[https://docs.zilliz.com.cn/docs/create-cluster]<br>

评论：不太好用