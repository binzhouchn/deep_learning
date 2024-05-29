from typing import Dict, List, Optional, Tuple, Union

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


class BaseModel:
    def __init__(self, path: str = '') -> None:
        self.path = path

    def chat(self, prompt: str, history: List[dict]):
        pass

    def load_model(self):
        pass

class OpenAIChat(BaseModel):
    def __init__(self, path: str = '', model: str = "glm-4") -> None:
        from openai import OpenAI
        super().__init__(path)
        self.model = model
        self.client = OpenAI(
            api_key = "ee748868eb25227724f960e69904f7b2.xsEtVM7S07AqFK9n",
            base_url = "https://open.bigmodel.cn/api/paas/v4/"
        )
    def chat(self, prompt: str, history: List[dict], meta_instruction:str ='') -> str:
        history.append({'role': 'system', 'content': meta_instruction})
        history.append({'role': 'user', 'content': prompt})
        res = self.client.chat.completions.create(
            model=self.model,
            messages=history,
            top_p=0.7,
            temperature=0.1
        )
        response = res.choices[0].message.content
        history.append({'role': 'assistant', 'content': response})
        return response, history

class InternLM2Chat(BaseModel):
    def __init__(self, path: str = '') -> None:
        super().__init__(path)
        self.load_model()

    def load_model(self):
        print('================ Loading model ================')
        self.tokenizer = AutoTokenizer.from_pretrained(self.path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(self.path, torch_dtype=torch.float16, trust_remote_code=True).to('mps').eval()
        print('================ Model loaded ================')

    def chat(self, prompt: str, history: List[dict], meta_instruction:str ='') -> str:
        response, history = self.model.chat(self.tokenizer, prompt, history, temperature=0.1, meta_instruction=meta_instruction)
        return response, history
    
# if __name__ == '__main__':
    # model = InternLM2Chat('/Users/zhoubin/pretrained/internlm2-chat-7b')
    # model = OpenAIChat() #推荐用接口方式
    # print(model.chat('你背后用的是什么模型啊', []))