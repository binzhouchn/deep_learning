# bert4keras

[**苏神实现的bert4keras**](#苏神实现的bert4keras)

[**苏神实现的simbert**](#苏神实现的simbert)





---

### 苏神实现的bert4keras

[网址](https://kexue.fm/archives/6915)<br>
```python
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer
import numpy as np

config_path = '../../kg/bert/chinese_L-12_H-768_A-12/bert_config.json'
checkpoint_path = '../../kg/bert/chinese_L-12_H-768_A-12/bert_model.ckpt'
dict_path = '../../kg/bert/chinese_L-12_H-768_A-12/vocab.txt'

tokenizer = Tokenizer(dict_path) # 建立分词器
model = build_transformer_model(config_path, checkpoint_path) # 建立模型，加载权重

# 编码测试
token_ids, segment_ids = tokenizer.encode(u'语言模型')
print(model.predict([np.array([token_ids]), np.array([segment_ids])]))
```

### 苏神实现的simbert

[simbert模型文件下载地址](https://github.com/Jie-Yuan/Pre-trainedModelZoo)<br>
[DEMO代码](simbert.py)
