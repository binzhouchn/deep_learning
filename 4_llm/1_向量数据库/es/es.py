#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
from tqdm import tqdm
from elasticsearch7 import Elasticsearch
from elasticsearch7 import helpers
#实例化ES
es = Elasticsearch("http://0.0.0.0:9200", timeout=6000, max_retries=3, retry_on_timeout=True)
# Call an API, in this example `info()`
print(es.info())

index_name = 'modelscope_fintech' #类似表名

request_body = {
    'mappings': {
        'properties': {
            'company': {
                'type': 'keyword' #在创建映射的时候，直接对不需要分词的字段设置keyword属性
            },
            'year': {
                'type': 'keyword'
            },
            'content':{
                'type': 'text',
                'analyzer': 'index_analyzer',
                'search_analyzer': 'small_search_analyzer',
                'fields': {
                    'kwd':{
                        'ignore_above': 100,
                        'type': 'keyword'
                    },
                    'suggest': {
                        'type': 'completion',
                        'analyzer': 'ik_max_word_pinyin'
                    }
                }
            },
            "vector": {
                        "type": "dense_vector",
                        "dims":4
                      },
            'id': {
                'type': 'integer'
            },  
        }
    },
    'settings': {
		"index": {
# 			"sort": {
# 				"field": "createDate",
# 				"order": "desc"
# 			},
			"store": {
				"preload": [
					"nvd",
					"dvd",
					"tim",
					"doc",
					"dim"
				]
			},
			"analysis": {
				"filter": {
					"pinyin_filter": {
						"lowercase": "true",
						"keep_original": "true",
						"remove_duplicated_term": "true",
						"keep_separate_first_letter": "false",
						"type": "pinyin",
						"limit_first_letter_length": "50",
						"keep_full_pinyin": "true"
					},
					"chinese_stopword": {
						"ignore_case": "true",
						"type": "stop",
						"stopwords_path": "stopwords/extra_stopword.dic"
					},
					"synonym_filter": {
						"type": "synonym",
						"synonyms_path": "synonyms/synonyms.dic"
					}
				},
				"analyzer": {
					"one_ngram_analyzer": {
						"filter": [
							"lowercase"
						],
						"type": "custom",
						"tokenizer": "one_ngram_tokenizer"
					},
					"small_index_analyzer": {
						"filter": [
							"lowercase"
						],
						"char_filter": [
							"html_strip"
						],
						"type": "custom",
						"tokenizer": "my_small_index_token"
					},
					"ik_max_word_pinyin": {
						"filter": [
							"pinyin_filter",
							"word_delimiter"
						],
						"type": "custom",
						"tokenizer": "ik_max_word"
					},
					"ik_smart_pinyin": {
						"filter": [
							"pinyin_filter",
							"word_delimiter"
						],
						"type": "custom",
						"tokenizer": "ik_smart"
					},
					"search_analyzer": {
						"filter": [
							"lowercase",
							"chinese_stopword",
							"synonym_filter"
						],
						"char_filter": [
							"html_strip"
						],
						"type": "custom",
						"tokenizer": "my_search_token"
					},
					"index_analyzer": {
						"filter": [
							"lowercase",
							"chinese_stopword",
							"synonym_filter"
						],
						"char_filter": [
							"html_strip"
						],
						"type": "custom",
						"tokenizer": "my_index_token"
					},
					"small_search_analyzer": {
						"filter": [
							"lowercase",
							"chinese_stopword",
							"synonym_filter"
						],
						"char_filter": [
							"html_strip"
						],
						"type": "custom",
						"tokenizer": "my_small_search_token"
					},
					"two_ngram_analyzer": {
						"filter": [
							"lowercase"
						],
						"type": "custom",
						"tokenizer": "two_ngram_tokenizer"
					}
				},
				"tokenizer": {
					"my_small_index_token": {
						"autoWordLength": "-1",
						"enableFailDingMsg": "true",
						"type": "hao_index_mode",
						"enableFallBack": "true",
						"enableSingleWord": "true"
					},
					"my_search_token": {
						"autoWordLength": "-1",
						"enableFailDingMsg": "true",
						"type": "hao_search_mode",
						"enableFallBack": "true",
						"enableSingleWord": "false"
					},
					"two_ngram_tokenizer": {
						"type": "ngram",
						"min_gram": "2",
						"max_gram": "2"
					},
					"my_small_search_token": {
						"autoWordLength": "-1",
						"enableFailDingMsg": "true",
						"type": "hao_search_mode",
						"enableFallBack": "true",
						"enableSingleWord": "true"
					},
					"my_index_token": {
						"autoWordLength": "-1",
						"enableFailDingMsg": "true",
						"type": "hao_index_mode",
						"enableFallBack": "true",
						"enableSingleWord": "false"
					},
					"one_ngram_tokenizer": {
						"type": "ngram",
						"min_gram": "1",
						"max_gram": "1"
					}
				}
			}
		}
	}
}

# 索引存在，先删除索引
if es.indices.exists(index_name):
    es.indices.delete(index=index_name)
    print('索引已删除，可重建！')
else:
    print('索引不存在，可以创建！')

# 创建索引
es.indices.create(index=index_name, body=request_body)

if __name__ == '__main__':
	#插入数据 - 单条插入
	es.index(index=index_name, id='1', body={
            'company': '国光电器股份有限公司',
            'year': '2021年'
            'content': '本集团是以出口为主的企业，以外币结算的海外销售占集团销售收入的85%以上，\
                        为规避主营业务中所产生的汇率波动风险，管理层根据汇率变化有计划地进行外汇衍生品交易业务。',
            'vector': [1.6,2,3.9,4.9],
            'id': 1,
        }
    )
    #插入数据 - 批量插入
	data_lst = [{
            'company': '国光电器股份有限公司',
            'year': '2021年'
            'content': '本集团是以出口为主的企业，以外币结算的海外销售占集团销售收入的85%以上，\
                        为规避主营业务中所产生的汇率波动风险，管理层根据汇率变化有计划地进行外汇衍生品交易业务。',
            'vector': [1.6,2,3.9,4.9],
            'id': 0,
        },
        {
            'company': '上海凯众材料科技股份有限公司',
            'content':'公司主营业务所属行业为汽车零部件制造业，主要从事汽车(涵盖传统汽车、新能源汽车和智能驾驶汽车)底盘悬架系统减震元件、操控系统轻量化踏板总成和电子驻车制动系统产品的研发、生产和销售，\
                        以及非汽车零部件领域高性能聚氨酯承载轮等特种聚氨酯弹性体的研发、生产和销售。',
            'year': '2019年',
            'vector': [1.72,2.0,4.0,5.0],
            'id': 1,
        }
	]
	#组装成actions
	actions = []
	for idx, p in enumerate(data_lst):
		actions.append({'_index': index_name,
            '_type':'_doc',
            '_id': idx,
            '_source':p})

	helpers.bulk(es, actions)


	#########################查询#########################
	#这次查询是完全匹配公司名，以及content中有销售关键词的召回出来后，根据向量cos值排序
	body = {#'explain':True,
	        'size':5,#返回top5
	        'query':
	                {
	                  "script_score": {
	                    "query": {
	                        'bool': {
	                            'must': [
	                                {
	                                    'term': {
	                                        'company': '国光电器股份有限公司'，
	                                        #'year': '2019年'
	                                    },
	                                },
	                                {
	                                    'match': {
	                                        'content':'销售'
	                                    }
	                                }
	                            ]    
	                        }    
	                    },
	                    "script": {
	                      "source": "cosineSimilarity(params.query_vector, 'vector') + 1.0",
	                      "params": {"query_vector": [1.7,2.0,4.0,5.0]}
	                    }
	                  }
	                }
	}
	es.search(index=index_name, body=body)
