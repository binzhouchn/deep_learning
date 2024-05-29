'''
仿照RAG/Embeddings.py的写法
'''
import os
import numpy as np
import requests
from typing import Dict, List, Optional, Tuple, Union

class BaseEmbeddings:
    """
    Base class for embeddings
    """
    def __init__(self, path: str, is_api: bool) -> None:
        self.path = path
        self.is_api = is_api
    
    def get_embedding(self, text: str, model: str) -> List[float]:
        raise NotImplementedError
    
    @classmethod
    def cosine_similarity(cls, vector1: List[float], vector2: List[float]) -> float:
        """
        calculate cosine similarity between two vectors
        """
        dot_product = np.dot(vector1, vector2)
        magnitude = np.linalg.norm(vector1) * np.linalg.norm(vector2)
        if not magnitude:
            return 0
        return dot_product / magnitude
    
class Bgem3Embedding(BaseEmbeddings):
    """
    class for bge-m3 embeddings
    """
    def __init__(self, path: str = '', is_api: bool = True) -> None:
        super().__init__(path, is_api)
        if self.is_api:
            self.url = "http://127.0.0.1:8001/v1/embeddings"
            self.headers = {
                "Content-Type": "application/json",
                "Authorization": "Bearer no-key",
                "content-type": "application/json"
            }
    
    def get_embedding(self, text: str, model: str = "bge-m3") -> List[float]:
        if self.is_api:
            payload = {
                "input": text,
                "model": "bge-m3",
                "encoding_format": "float"
            }
            if isinstance(text, str):
                text = text.replace("\n", " ")
            else:
                raise ValueError("请检查text格式，是否是string！")
            res = requests.request("POST", self.url, json=payload, headers=self.headers).json()
            return res["data"][0]["embedding"]
        else:
            raise NotImplementedError
        
class JinaEmbedding(BaseEmbeddings):
    """
    class for Jina embeddings
    """
    def __init__(self, path: str = './pretrained/jina-embeddings-v2-base-zh', is_api: bool = False) -> None:
        super().__init__(path, is_api)
        self._model = self.load_model()
        
    def get_embedding(self, text: str) -> List[float]:
        return self._model.encode([text])[0].tolist()
    
    def load_model(self):
        import torch
        from transformers import AutoModel
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        model = AutoModel.from_pretrained(self.path, trust_remote_code=True).to(device)
        return model

if __name__ == "__main__":
    bge = Bgem3Embedding()
    a = bge.get_embedding("中国")
    print(a)
    