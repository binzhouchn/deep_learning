# https://github.com/datawhalechina/wow-agent
# https://github.com/datawhalechina/wow-rag

#配置llm
#配置embedding
import os
import sys
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()
# 从环境变量中读取api_key，glm模型
api_key = 'sk-CheAFXpUwl7euofb660f9269E3Af4609BdFfF426E57d46A5'
base_url = "http://43.200.7.56:8008/v1"
chat_model = "glm-4-flash"
emb_model = "embedding-3"
from openai import OpenAI
from pydantic import Field  # 导入Field，用于Pydantic模型中定义字段的元数据
from llama_index.core.llms import (
    CustomLLM,
    CompletionResponse,
    LLMMetadata,
)
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.llms.callbacks import llm_completion_callback
from typing import List, Any, Generator
# 定义OurLLM类，继承自CustomLLM基类
class OurLLM(CustomLLM):
    api_key: str = Field(default=api_key)
    base_url: str = Field(default=base_url)
    model_name: str = Field(default=chat_model)
    client: OpenAI = Field(default=None, exclude=True)  # 显式声明 client 字段

    def __init__(self, api_key: str, base_url: str, model_name: str = chat_model, **data: Any):
        super().__init__(**data)
        self.api_key = api_key
        self.base_url = base_url
        self.model_name = model_name
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)  # 使用传入的api_key和base_url初始化 client 实例

    @property
    def metadata(self) -> LLMMetadata:
        """Get LLM metadata."""
        return LLMMetadata(
            model_name=self.model_name,
        )

    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        response = self.client.chat.completions.create(model=self.model_name, messages=[{"role": "user", "content": prompt}])
        if hasattr(response, 'choices') and len(response.choices) > 0:
            response_text = response.choices[0].message.content
            return CompletionResponse(text=response_text)
        else:
            raise Exception(f"Unexpected response format: {response}")

    @llm_completion_callback()
    def stream_complete(
        self, prompt: str, **kwargs: Any
    ) -> Generator[CompletionResponse, None, None]:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            stream=True
        )

        try:
            for chunk in response:
                chunk_message = chunk.choices[0].delta
                if not chunk_message.content:
                    continue
                content = chunk_message.content
                yield CompletionResponse(text=content, delta=content)

        except Exception as e:
            raise Exception(f"Unexpected response format: {e}")

class OurEmbeddings(BaseEmbedding):
    api_key: str = Field(default=api_key)
    base_url: str = Field(default=base_url)
    model_name: str = Field(default=emb_model)
    client: OpenAI = Field(default=None, exclude=True)  # 显式声明 client 字段

    def __init__(
        self,
        api_key: str = api_key, 
        base_url: str = base_url,
        model_name: str = emb_model,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.api_key = api_key
        self.base_url = base_url
        self.model_name = model_name
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def invoke_embedding(self, query: str) -> List[float]:
        response = self.client.embeddings.create(model=self.model_name, input=[query])

        # 检查响应是否成功
        if response.data and len(response.data) > 0:
            return response.data[0].embedding
        else:
            raise ValueError("Failed to get embedding from ZhipuAI API")

    def _get_query_embedding(self, query: str) -> List[float]:
        return self.invoke_embedding(query)

    def _get_text_embedding(self, text: str) -> List[float]:
        return self.invoke_embedding(text)

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        return [self._get_text_embedding(text) for text in texts]

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return self._get_query_embedding(query)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        return self._get_text_embedding(text)

    async def _aget_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        return self._get_text_embeddings(texts)

llm = OurLLM(api_key=api_key, base_url=base_url, model_name=chat_model)
embedding = OurEmbeddings(api_key=api_key, base_url=base_url, model_name=emb_model)

if __name__ == "__main__":
    #测试llm和embedding
    response = llm.complete("你背后用的啥模型啊")
    print(response)
    print('####################')
    emb = embedding.get_text_embedding("你好呀呀")
    print(len(emb), type(emb))