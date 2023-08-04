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

## 2. Milvus












## 3. pgvector
