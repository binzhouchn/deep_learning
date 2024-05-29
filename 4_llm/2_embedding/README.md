# embedding

## 1.bge-m3

```shell
# 我这边下载的guff格式https://huggingface.co/vonjack/bge-m3-gguf，本地装好llama.cpp后启动服务
./server -m ./user/bge-m3-f16.gguf --embedding -c 8192 --host 0.0.0.0 --port 8001
```
```python
# 然后python代码调用
import requests
from sentence_transformers import util as st_util
import numpy as np
def get_embeding(text):
    url = "http://127.0.0.1:8001/v1/embeddings"
    payload = {
        "input": text,
        "model": "GPT-4",
        "encoding_format": "float"
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer no-key",
        "content-type": "application/json"
    }
    response = requests.request("POST", url, json=payload, headers=headers)
    return response.json()

if __name__ == "__main__":
    a = get_embeding("中国")
    a_embedding = a["data"][0]["embedding"]
    print(len(a_embedding))
    b = get_embeding("中华人民共和国")
    b_embedding = b["data"][0]["embedding"]
    temp = st_util.cos_sim(np.array(a_embedding), np.array(b_embedding))
    print(temp)
    print(np.array(a_embedding) @ np.array(b_embedding))
```