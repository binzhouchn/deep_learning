# Copyright (c) OpenMMLab. All rights reserved.
"""extract feature and search with user query."""
import os
import time

import numpy as np
import pytoml
from BCEmbedding.tools.langchain import BCERerank
from langchain.retrievers import ContextualCompressionRetriever
from langchain.vectorstores.faiss import FAISS as Vectorstore
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.utils import DistanceStrategy
from loguru import logger
from sklearn.metrics import precision_recall_curve

from file_operation import FileOperation


class Retriever:
    """Tokenize and extract features from the project's documents, for use in
    the reject pipeline and response pipeline."""

    def __init__(self, embeddings, reranker, work_dir: str) -> None:
        """Init with model device type and config."""
        self.retriever = None
        self.compression_retriever = None

        if not os.path.exists(work_dir):
            logger.warning('!!!warning, workdir not exist.!!!')
            return

        retriever_path = os.path.join(work_dir, 'db_response')

        if not os.path.exists(retriever_path):
            logger.warning('!!!warning, feature db not exist.!!!')
            return

        self.retriever = Vectorstore.load_local(
            retriever_path,
            embeddings=embeddings,
            allow_dangerous_deserialization=True,
            distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT).as_retriever(
                search_type='similarity',
                search_kwargs={
                    'score_threshold': 0.15,
                    'k': 30
                })
        self.compression_retriever = ContextualCompressionRetriever(
            base_compressor=reranker, base_retriever=self.retriever)

    def query(self,
              question: str,
              context_max_length: int = 16000):
        """Processes a query and returns the best match from the vector store
        database. If the question is rejected, returns None.

        Args:
            question (str): The question asked by the user.

        Returns:
            str: The best matching chunk, or None.
            str: The best matching text, or None
        """
        if question is None or len(question) < 1:
            return None, None, []

        if len(question) > 512:
            logger.warning('input too long, truncate to 512')
            question = question[0:512]

        chunks = []
        context = ''
        references = []

        docs = self.compression_retriever.get_relevant_documents(question)
        # add file text to context, until exceed `context_max_length`

        file_opr = FileOperation()
        for idx, doc in enumerate(docs):
            chunk = doc.page_content
            chunks.append(chunk)

            if 'read' not in doc.metadata:
                logger.error(
                    'If you are using the version before 20240319, please rerun `python3 -m huixiangdou.service.feature_store`'
                )
                raise Exception('huixiangdou version mismatch')
            file_text, error = file_opr.read(doc.metadata['read'])
            if error is not None:
                # read file failed, skip
                continue

            source = doc.metadata['source']
            logger.info('target {} file length {}'.format(
                source, len(file_text)))
            if len(file_text) + len(context) > context_max_length:
                if source in references:
                    continue
                references.append(source)
                # add and break
                add_len = context_max_length - len(context)
                if add_len <= 0:
                    break
                chunk_index = file_text.find(chunk)
                if chunk_index == -1:
                    # chunk not in file_text
                    context += chunk
                    context += '\n'
                    context += file_text[0:add_len - len(chunk) - 1]
                else:
                    start_index = max(0, chunk_index - (add_len - len(chunk)))
                    context += file_text[start_index:start_index + add_len]
                break

            if source not in references:
                context += file_text
                context += '\n'
                references.append(source)

        context = context[0:context_max_length]
        logger.debug('query:{} top1 file:{}'.format(question, references[0]))
        return '\n'.join(chunks), context, [
            os.path.basename(r) for r in references
        ]


class CacheRetriever:

    def __init__(self, config_path: str, max_len: int = 4):
        self.cache = dict()
        self.max_len = max_len
        with open(config_path, encoding='utf8') as f:
            config = pytoml.load(f)['feature_store']
            embedding_model_path = config['embedding_model_path']
            reranker_model_path = config['reranker_model_path']

        # load text2vec and rerank model
        logger.info('loading test2vec and rerank models')
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model_path,
            model_kwargs={'device': 'mps'},
            encode_kwargs={
                'batch_size': 1,
                'normalize_embeddings': True
            })
        self.embeddings.client = self.embeddings.client.half()
        reranker_args = {
            'model': reranker_model_path,
            'top_n': 5,
            'device': 'mps',
            'use_fp16': True
        }
        self.reranker = BCERerank(**reranker_args)

    def get(self,
            fs_id: str = 'default',
            config_path='config.ini',
            work_dir='workdir'):
        if fs_id in self.cache:
            self.cache[fs_id]['time'] = time.time()
            return self.cache[fs_id]['retriever']

        if len(self.cache) >= self.max_len:
            # drop the oldest one
            del_key = None
            min_time = time.time()
            for key, value in self.cache.items():
                cur_time = value['time']
                if cur_time < min_time:
                    min_time = cur_time
                    del_key = key

            if del_key is not None:
                del_value = self.cache[del_key]
                self.cache.pop(del_key)
                del del_value['retriever']

        retriever = Retriever(embeddings=self.embeddings,
                              reranker=self.reranker,
                              work_dir=work_dir)
        self.cache[fs_id] = {'retriever': retriever, 'time': time.time()}
        return retriever

    def pop(self, fs_id: str):
        if fs_id not in self.cache:
            return
        del_value = self.cache[fs_id]
        self.cache.pop(fs_id)
        # manually free memory
        del del_value
