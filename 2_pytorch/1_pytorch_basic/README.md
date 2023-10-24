# pytorch实用技巧

[**1. 得到模型参数数量**](#得到模型参数数量)

[**2. 特定网络结构参数分布初始化**](#特定网络结构参数分布初始化)

[**3. view函数**](#view函数)

[**4. unsqueeze函数**](#unsqueeze函数)

[**5. squeeze函数**](#squeeze函数)

[**6. pytorch自定义损失函数**](#pytorch自定义损失函数)

[**7. pytorch自定义矩阵W**](#pytorch自定义矩阵w)

[**8. 自定义操作torch.autograd.Function**](#autograd)

[**9. pytorch embedding设置不可导**](#pytorch_embedding设置不可导)

[**10. 中文tokenizer**](#中文tokenizer)

[**11. Accelerate: 适用于多GPU、TPU、混合精度训练**](#accelerate)

[**12. pytorch删除一层网络**](#pytorch删除一层网络)

[**13. Focal Loss with alpha**](#focal_loss)

[**14. 清空显存方法**](#清空显存)

---

## 得到模型参数数量

```python
def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}
get_parameter_number(model)
```

## 特定网络结构参数分布初始化

```python
class AutoEncoder(nn.Module):
    def __init__(self, feedback_bits):
        super(AutoEncoder, self).__init__()
        self.encoder = Encoder(feedback_bits)
        self.decoder = Decoder(feedback_bits)
        ###-------初始化参数分布------###
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        ###------------------------###
    def forward(self, x):
        feature = self.encoder(x)
        out = self.decoder(feature)
        return out
```

## view函数

```python
import torch as t
a = t.arange(0,6)
print(a)
# tensor([ 0.,  1.,  2.,  3.,  4.,  5.])
a.view(2,-1) # 行数为2行，-1表示列自动计算
# tensor([[ 0.,  1.,  2.],
#         [ 3.,  4.,  5.]])
```

## unsqueeze函数

```python
import torch as t
a = t.arange(0,6).view(2,3)
print(a)
# tensor([[ 0.,  1.,  2.],
#         [ 3.,  4.,  5.]])
print(a.size())
# torch.Size([2, 3])
```
```python
# 有点像reshape
a.unsqueeze(0).size()
# torch.Size([1, 2, 3])
a.unsqueeze(1).size()
# torch.Size([2, 1, 3])
a.unsqueeze(2).size()
# torch.Size([2, 3, 1])
```

## squeeze函数

```python
import torch
a = torch.Tensor([[1,2,3]])
# tensor([[1., 2., 3.]])
a.squeeze()
# tensor([1., 2., 3.])
a = torch.Tensor([1,2,3,4,5,6])
a.view(2,3)
# tensor([[1., 2., 3.],
#        [4., 5., 6.]])
a.squeeze()
# tensor([1., 2., 3., 4., 5., 6.])
```

## pytorch自定义损失函数

![nwrmsle.png](pic/nwrmsle.png)

```python
# pytorch自定义损失函数 Normalized Weighted Root Mean Squared Logarithmic Error(NWRMSLE)
# 这里y真实值需要提前进行log1p的操作
# 加入了sample_weights，和keras里model.fit(x,sample_weights)一样
from torch.functional import F

class my_rmseloss(nn.Module):
    
    def __init__(self):
        super(my_rmseloss, self).__init__()
        return 
    
    def forward(self, input, target, sample_weights=None):
        self._assert_no_grad(target)
        f_revis = lambda a, b, w: ((a - b) ** 2) * w # 重写
        return self._pointwise_loss(f_revis, torch._C._nn.mse_loss,
                           input, target, sample_weights)
    
    # 重写_pointwise_loss
    def _pointwise_loss(self, lambd, lambd_optimized, input, target, sample_weights):
        if target.requires_grad:
            d = lambd(input, target, sample_weights)
            return torch.sqrt(torch.div(torch.sum(d), torch.sum(sample_weights)))
        else:
            if sample_weights is not None:
                unrooted_res = torch.div(torch.sum(torch.mul(lambd_optimized(input, target),sample_weights)),torch.sum(sample_weights))
                return torch.sqrt(unrooted_res)
            return lambd_optimized(input, target, 1)
    
    def _assert_no_grad(self, tensor):
        assert not tensor.requires_grad, \
            "nn criterions don't compute the gradient w.r.t. targets - please " \
            "mark these tensors as not requiring gradients"
```

### pytorch自定义矩阵w

比如我在DigitCaps中定义了一个W的矩阵，想要这个矩阵可导，则用nn.Parameter包一下

```python
class DigitCaps(nn.Module):
    def __init__(self, num_capsules=10, num_routes=32 * 40, in_channels=10, out_channels=16):
        super(DigitCaps, self).__init__()

        self.in_channels = in_channels
        self.num_routes = num_routes
        self.num_capsules = num_capsules

        self.W = nn.Parameter(torch.randn(1, num_routes, num_capsules, out_channels, in_channels),requires_grad=True) # 可导

    def forward(self, x):
        batch_size = x.size(0)
        x = torch.stack([x] * self.num_capsules, dim=2).unsqueeze(4)

        W = torch.cat([self.W] * batch_size, dim=0)
        u_hat = torch.matmul(W, x)
```
```python
# 把上面的加载进优化器就行了，如果这个DigitCaps在其他类中被调用，则
# 把最初始的那个main类加载入Adam就行
dcaps = DigitCaps()
optimizer = Adam(dcaps.parameters(),lr=0.001)
```

[和Keras build里面的self.add_weight是一样的](https://keras.io/zh/layers/writing-your-own-keras-layers/)

## autograd

[PyTorch 74.自定义操作torch.autograd.Function - 讲的很好](https://zhuanlan.zhihu.com/p/344802526)

### pytorch_embedding设置不可导

```python
self.encoder.weight = nn.Parameter(t.from_numpy(embedding_matrix).float(), requires_grad=False)
```

### 中文tokenizer

```python
import six
import unicodedata


def convert_to_unicode(text):
    """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
    if six.PY3:
        if isinstance(text, str):
            return text
        elif isinstance(text, bytes):
            return text.decode("utf-8", "ignore")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    elif six.PY2:
        if isinstance(text, str):
            return text.decode("utf-8", "ignore")
        elif isinstance(text, unicode):
            return text
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    else:
        raise ValueError("Not running on Python2 or Python 3?")


class BasicTokenizer(object):
    """Runs basic tokenization (punctuation splitting, lower casing, etc.)."""

    def __init__(self, do_lower_case=True):
        """Constructs a BasicTokenizer.
    
        Args:
          do_lower_case: Whether to lower case the input.
        """
        self.do_lower_case = do_lower_case

    def tokenize(self, text):
        """Tokenizes a piece of text."""
        text = convert_to_unicode(text)
        text = self._clean_text(text)

        # This was added on November 1st, 2018 for the multilingual and Chinese
        # models. This is also applied to the English models now, but it doesn't
        # matter since the English models were not trained on any Chinese data
        # and generally don't have any Chinese data in them (there are Chinese
        # characters in the vocabulary because Wikipedia does have some Chinese
        # words in the English Wikipedia.).
        text = self._tokenize_chinese_chars(text)

        orig_tokens = whitespace_tokenize(text)
        split_tokens = []
        for token in orig_tokens:
            if self.do_lower_case:
                token = token.lower()
                token = self._run_strip_accents(token)
            split_tokens.extend(self._run_split_on_punc(token))

        output_tokens = whitespace_tokenize(" ".join(split_tokens))
        return output_tokens

    def _run_strip_accents(self, text):
        """Strips accents from a piece of text."""
        text = unicodedata.normalize("NFD", text)
        output = []
        for char in text:
            cat = unicodedata.category(char)
            if cat == "Mn":
                continue
            output.append(char)
        return "".join(output)

    def _run_split_on_punc(self, text):
        """Splits punctuation on a piece of text."""
        chars = list(text)
        i = 0
        start_new_word = True
        output = []
        while i < len(chars):
            char = chars[i]
            if _is_punctuation(char):
                output.append([char])
                start_new_word = True
            else:
                if start_new_word:
                    output.append([])
                start_new_word = False
                output[-1].append(char)
            i += 1

        return ["".join(x) for x in output]

    def _tokenize_chinese_chars(self, text):
        """Adds whitespace around any CJK character."""
        output = []
        for char in text:
            cp = ord(char)
            if self._is_chinese_char(cp):
                output.append(" ")
                output.append(char)
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)

    def _is_chinese_char(self, cp):
        """Checks whether CP is the codepoint of a CJK character."""
        # This defines a "chinese character" as anything in the CJK Unicode block:
        #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
        #
        # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
        # despite its name. The modern Korean Hangul alphabet is a different block,
        # as is Japanese Hiragana and Katakana. Those alphabets are used to write
        # space-separated words, so they are not treated specially and handled
        # like the all of the other languages.
        if ((cp >= 0x4E00 and cp <= 0x9FFF) or  #
                (cp >= 0x3400 and cp <= 0x4DBF) or  #
                (cp >= 0x20000 and cp <= 0x2A6DF) or  #
                (cp >= 0x2A700 and cp <= 0x2B73F) or  #
                (cp >= 0x2B740 and cp <= 0x2B81F) or  #
                (cp >= 0x2B820 and cp <= 0x2CEAF) or
                (cp >= 0xF900 and cp <= 0xFAFF) or  #
                (cp >= 0x2F800 and cp <= 0x2FA1F)):  #
            return True

        return False

    def _clean_text(self, text):
        """Performs invalid character removal and whitespace cleanup on text."""
        output = []
        for char in text:
            cp = ord(char)
            if cp == 0 or cp == 0xfffd or _is_control(char):
                continue
            if _is_whitespace(char):
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)


def _is_whitespace(char):
    """Checks whether `chars` is a whitespace character."""
    # \t, \n, and \r are technically contorl characters but we treat them
    # as whitespace since they are generally considered as such.
    if char == " " or char == "\t" or char == "\n" or char == "\r":
        return True
    cat = unicodedata.category(char)
    if cat == "Zs":
        return True
    return False


def _is_control(char):
    """Checks whether `chars` is a control character."""
    # These are technically control characters but we count them as whitespace
    # characters.
    if char == "\t" or char == "\n" or char == "\r":
        return False
    cat = unicodedata.category(char)
    if cat in ("Cc", "Cf"):
        return True
    return False


def _is_punctuation(char):
    """Checks whether `chars` is a punctuation character."""
    cp = ord(char)
    # We treat all non-letter/number ASCII as punctuation.
    # Characters such as "^", "$", and "`" are not in the Unicode
    # Punctuation class but we treat them as punctuation anyways, for
    # consistency.
    if ((cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or
            (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126)):
        return True
    cat = unicodedata.category(char)
    if cat.startswith("P"):
        return True
    return False


def whitespace_tokenize(text):
    """Runs basic whitespace cleaning and splitting on a piece of text."""
    text = text.strip()
    if not text:
        return []
    tokens = text.split()
    return tokens


if __name__ == '__main__':
    a = BasicTokenizer()
    print(a.tokenize("我是Im chinese中国人hhh啊~"))

```

### accelerate

[Hugging Face发布PyTorch新库「Accelerate」：适用于多GPU、TPU、混合精度训练](https://mp.weixin.qq.com/s/-AjNv3E7NUkGGIruAm_SvQ)<br>
```python
# +为代码增加项，-为减项
import torch
  import torch.nn.functional as F
  from datasets import load_dataset
+ from accelerate import Accelerator

+ accelerator = Accelerator()
- device = 'cpu'
+ device = accelerator.device

  model = torch.nn.Transformer().to(device)
  optim = torch.optim.Adam(model.parameters())

  dataset = load_dataset('my_dataset')
  data = torch.utils.data.DataLoader(dataset, shuffle=True)

+ model, optim, data = accelerator.prepare(model, optim, data)

  model.train()
  for epoch in range(10):
      for source, targets in data:
          source = source.to(device)
          targets = targets.to(device)

          optimizer.zero_grad()

          output = model(source)
          loss = F.cross_entropy(output, targets)

-         loss.backward()
+         accelerator.backward(loss)

          optimizer.step()
```

### pytorch删除一层网络

删除最后一层为例<br>
```python
from sentence_transformers import SentenceTransformer, util
from torch import nn
def load_tranformer_model():
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('distiluse-base-multilingual-cased-v1') #distilbert-base-nli-stsb-mean-tokens #distiluse-base-multilingual-cased-v1
    return model
model = load_tranformer_model()
print(model)
'''
SentenceTransformer(
  (0): Transformer({'max_seq_length': 128, 'do_lower_case': False}) with Transformer model: DistilBertModel 
  (1): Pooling({'word_embedding_dimension': 768, 'pooling_mode_cls_token': False, 'pooling_mode_mean_tokens': True, 'pooling_mode_max_tokens': False, 'pooling_mode_mean_sqrt_len_tokens': False})
  (2): Dense({'in_features': 768, 'out_features': 512, 'bias': True, 'activation_function': 'torch.nn.modules.activation.Tanh'})
)
'''
# 现在我想去掉最后一层Dense层
new_model = SentenceTransformer(modules=list(model.children())[:-1])
print(new_model)
'''
SentenceTransformer(
  (0): Transformer({'max_seq_length': 128, 'do_lower_case': False}) with Transformer model: DistilBertModel 
  (1): Pooling({'word_embedding_dimension': 768, 'pooling_mode_cls_token': False, 'pooling_mode_mean_tokens': True, 'pooling_mode_max_tokens': False, 'pooling_mode_mean_sqrt_len_tokens': False})
)
'''
```

修改最后一层<br
比如最后Dense层需要将out_features从512改成100，那么需要先拿到原模型Dense的类，然后修改下后<br>
```python
new_model = SentenceTransformer(modules=(list(model.children())[:-1]+[Dense2(768,100)]))
print(new_model)
'''
SentenceTransformer(
  (0): Transformer({'max_seq_length': 128, 'do_lower_case': False}) with Transformer model: DistilBertModel 
  (1): Pooling({'word_embedding_dimension': 768, 'pooling_mode_cls_token': False, 'pooling_mode_mean_tokens': True, 'pooling_mode_max_tokens': False, 'pooling_mode_mean_sqrt_len_tokens': False})
  (2): Dense({'in_features': 768, 'out_features': 100, 'bias': True, 'activation_function': 'torch.nn.modules.activation.Tanh'})
)

'''
```

### focal_loss

```python
import torch

class MultiClassFocalLossWithAlpha(nn.Module):
    def __init__(self, alpha=[0.2, 0.3, 0.5], gamma=2, reduction='mean'):
        """
        :param alpha: 权重系数列表，三分类中第0类权重0.2，第1类权重0.3，第2类权重0.5
        :param gamma: 困难样本挖掘的gamma
        :param reduction:
        """
        super(MultiClassFocalLossWithAlpha, self).__init__()
        self.alpha = torch.tensor(alpha).to(device)
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, pred, target):
        alpha = self.alpha[target]  # 为当前batch内的样本，逐个分配类别权重，shape=(bs), 一维向量
        log_softmax = torch.log_softmax(pred, dim=1) # 对模型裸输出做softmax再取log, shape=(bs, 3)
        logpt = torch.gather(log_softmax, dim=1, index=target.view(-1, 1))  # 取出每个样本在类别标签位置的log_softmax值, shape=(bs, 1)
        logpt = logpt.view(-1)  # 降维，shape=(bs)
        ce_loss = -logpt  # 对log_softmax再取负，就是交叉熵了
        pt = torch.exp(logpt)  #对log_softmax取exp，把log消了，就是每个样本在类别标签位置的softmax值了，shape=(bs)
        focal_loss = alpha * (1 - pt) ** self.gamma * ce_loss  # 根据公式计算focal loss，得到每个样本的loss值，shape=(bs)
        if self.reduction == "mean":
            return torch.mean(focal_loss)
        if self.reduction == "sum":
            return torch.sum(focal_loss)
        return focal_loss

#[star]训练,prob1+prob2mul保费多分类focal_loss
weights = [x/sum([1,5,5,5,5,200]) for x in [1,5,5,5,5,200]] #6类标签的权重，数量越少的标签设置权重越大
loss_fn = MultiClassFocalLossWithAlpha(alpha=weights).to(device)
loss_fn(pred,target) # 以batch_size=2为例 pred = [[0.2,0.4,0.1,0.15,0.13,0.02],[0.1,0.14,0.53,0.07,0.11,0.05]] target = [1,3]
```

### 清空显存

```python
def clean_memory():
    gc.collect()
    ctypes.CDLL("libc.so.6").malloc_trim(0)
    torch.cuda.empty_cache()
```

---

参考网址：

[pytorch中Liner、RNN、LSTM、RNN模型、输入和输出构造参数小结](https://blog.csdn.net/david0611/article/details/81090294)
