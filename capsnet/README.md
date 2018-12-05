# capsnet详解

[原论文](https://arxiv.org/pdf/1710.09829.pdf)

[先读懂CapsNet架构然后用TensorFlow实现：全面解析Hinton提出的Capsule](https://www.jiqizhixin.com/articles/2017-11-05)


1. PrimaryCapsLayer中的squash压缩的是向量size是[batch_size, 1152, 8]，在最后一个维度上进行压缩即维度8
压缩率|Sj|2/(1+|Sj|2)/|Sj|大小为[batch_size, 1152]，然后与原来的输入向量相乘即可

2. 如果reconstruction为True，则loss由两部分组成margin_loss和reconstruction_loss<br>
```python
output, probs = model(data, target)
reconstruction_loss = F.mse_loss(output, data.view(-1, 784))
margin_loss = loss_fn(probs, target)
# 如果reconstruction为True，则loss由两部分组成margin_loss和reconstruction_loss
loss = reconstruction_alpha * reconstruction_loss + margin_loss
```
