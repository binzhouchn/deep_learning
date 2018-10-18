# pytorch实用技巧

[**1. view函数**](#view函数)

[**2. unsqueeze函数**](#unsqueeze函数)

[**3. squeeze函数**](#squeeze函数)

[**4. pytorch自定义损失函数**](#pytorch自定义损失函数)

[**5. pytorch自定义矩阵W**](#pytorch自定义矩阵w)

---

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

        self.W = nn.Parameter(torch.randn(1, num_routes, num_capsules, out_channels, in_channels))

    def forward(self, x):
        batch_size = x.size(0)
        x = torch.stack([x] * self.num_capsules, dim=2).unsqueeze(4)

        W = torch.cat([self.W] * batch_size, dim=0)
        u_hat = torch.matmul(W, x)
```
```python
# 把上面的加载进优化器就行了
dcaps = DigitCaps()
optimizer = Adam(dcaps.parameters(),lr=0.001)
```



---

参考网址：

[pytorch中Liner、RNN、LSTM、RNN模型、输入和输出构造参数小结](https://blog.csdn.net/david0611/article/details/81090294)
