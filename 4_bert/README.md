# BERT笔记

[**1. bert网址**](#bert网址)

[**2. bert相关数据集下载**](#bert相关数据集下载)

[**3. 使用BERT和TensorFlow构建文本分类器**](#使用bert和tensorflow构建文本分类器)

[**4. bert跑中文任务**](#bert跑中文任务)

[**5. bert DIY**](#bert_diy)

[**6. 遇到的问题**](#遇到的问题)

[**7. XLNET相关**](#xlnet相关)

---

### bert网址

bert论文<br>
[BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)<br>
[图解BERT模型：从零开始构建BERT](https://cloud.tencent.com/developer/article/1389555)<br>

bert代码实现<br>
[github地址 tf版](https://github.com/google-research/bert)<br>
[github地址 bert-as-service](https://github.com/hanxiao/bert-as-service)<br>
[github地址 pytorch版](https://github.com/codertimo/BERT-pytorch)<br>

bert测试数据集<br>
[BERT之'测试数据集描述'](https://blog.csdn.net/shuibuzhaodeshiren/article/details/87743286)<br>
[预训练模型测试数据集PaddlePaddle](https://github.com/PaddlePaddle/ERNIE/blob/develop/README.zh.md#%E6%95%B0%E6%8D%AE%E4%B8%8B%E8%BD%BD)<br>

### bert相关数据集下载

```python
TASKS = ["CoLA", "SST", "MRPC", "QQP", "STS", "MNLI", "SNLI", "QNLI", "RTE", "WNLI", "diagnostic"]
DATA_FOR_TASKS = {"CoLA":'https://firebasestorage.googleapis.com/v0/b/mtl-sentence-representations.appspot.com/o/data%2FCoLA.zip?alt=media&token=46d5e637-3411-4188-bc44-5809b5bfb5f4',
             "SST":'https://firebasestorage.googleapis.com/v0/b/mtl-sentence-representations.appspot.com/o/data%2FSST-2.zip?alt=media&token=aabc5f6b-e466-44a2-b9b4-cf6337f84ac8',
             "MRPC":'https://firebasestorage.googleapis.com/v0/b/mtl-sentence-representations.appspot.com/o/data%2Fmrpc_dev_ids.tsv?alt=media&token=ec5c0836-31d5-48f4-b431-7480817f1adc',
             "QQP":'https://firebasestorage.googleapis.com/v0/b/mtl-sentence-representations.appspot.com/o/data%2FQQP.zip?alt=media&token=700c6acf-160d-4d89-81d1-de4191d02cb5',
             "STS":'https://firebasestorage.googleapis.com/v0/b/mtl-sentence-representations.appspot.com/o/data%2FSTS-B.zip?alt=media&token=bddb94a7-8706-4e0d-a694-1109e12273b5',
             "MNLI":'https://firebasestorage.googleapis.com/v0/b/mtl-sentence-representations.appspot.com/o/data%2FMNLI.zip?alt=media&token=50329ea1-e339-40e2-809c-10c40afff3ce',
             "SNLI":'https://firebasestorage.googleapis.com/v0/b/mtl-sentence-representations.appspot.com/o/data%2FSNLI.zip?alt=media&token=4afcfbb2-ff0c-4b2d-a09a-dbf07926f4df',
             "QNLI":'https://firebasestorage.googleapis.com/v0/b/mtl-sentence-representations.appspot.com/o/data%2FQNLI.zip?alt=media&token=c24cad61-f2df-4f04-9ab6-aa576fa829d0',
             "RTE":'https://firebasestorage.googleapis.com/v0/b/mtl-sentence-representations.appspot.com/o/data%2FRTE.zip?alt=media&token=5efa7e85-a0bb-4f19-8ea2-9e1840f077fb',
             "WNLI":'https://firebasestorage.googleapis.com/v0/b/mtl-sentence-representations.appspot.com/o/data%2FWNLI.zip?alt=media&token=068ad0a0-ded7-4bd7-99a5-5e00222e0faf',
             "diagnostic":'https://storage.googleapis.com/mtl-sentence-representations.appspot.com/tsvsWithoutLabels%2FAX.tsv?GoogleAccessId=firebase-adminsdk-0khhl@mtl-sentence-representations.iam.gserviceaccount.com&Expires=2498860800&Signature=DuQ2CSPt2Yfre0C%2BiISrVYrIFaZH1Lc7hBVZDD4ZyR7fZYOMNOUGpi8QxBmTNOrNPjR3z1cggo7WXFfrgECP6FBJSsURv8Ybrue8Ypt%2FTPxbuJ0Xc2FhDi%2BarnecCBFO77RSbfuz%2Bs95hRrYhTnByqu3U%2FYZPaj3tZt5QdfpH2IUROY8LiBXoXS46LE%2FgOQc%2FKN%2BA9SoscRDYsnxHfG0IjXGwHN%2Bf88q6hOmAxeNPx6moDulUF6XMUAaXCSFU%2BnRO2RDL9CapWxj%2BDl7syNyHhB7987hZ80B%2FwFkQ3MEs8auvt5XW1%2Bd4aCU7ytgM69r8JDCwibfhZxpaa4gd50QXQ%3D%3D'}

MRPC_TRAIN = 'https://s3.amazonaws.com/senteval/senteval_data/msr_paraphrase_train.txt'
MRPC_TEST = 'https://s3.amazonaws.com/senteval/senteval_data/msr_paraphrase_test.txt'

```

### 使用bert和tensorflow构建文本分类器

文本二分类任务，dataprocessor用cola<br>
```shell
python run_classifier.py --task_name=cola --do_train=true --do_eval=true --do_predict=true --data_dir=/home/zhoubin/bert/data/ --vocab_file=/home/zhoubin/bert/uncased_L-12_H-768_A-12/vocab.txt --bert_config_file=/home/zhoubin/bert/uncased_L-12_H-768_A-12/bert_config.json --init_checkpoint=/home/zhoubin/bert/uncased_L-12_H-768_A-12/bert_model.ckpt --max_seq_length=128 --train_batch_size=32 --learning_rate=2e-5 --num_train_epochs=5.0 --output_dir=/home/zhoubin/bert/output --do_lower_case=True
```

新生成的三个模型文件<br>
```shell
model.ckpt-781.data-00000-of-00001
model.ckpt-781.index
model.ckpt-781.meta
```

预测<br>


### bert跑中文任务



### bert_DIY

搞清楚run_classifier.py文件

自定义Processor，继承DataProcessor<br>
修改input_fn_builder<br>
修改model_fn_builder和里面的create_model，其中create_model中output_layer后面可以自己接其他的层比如CRF层然后再改一下loss的计算方式或者不改<br>
bert create_model中最后一层其实就是Dense层和log_loss，我们可以直接用封装好的Dense不需要像源码自己写

### 遇到的问题

1. 加载预训练权重到bert后，如果在train模式下预测同一个字则会得到不同的字向量，因为train模式下有dropout或者batchnorm；
所以pytorch下可以bert_embed.eval()(words)这样

### xlnet相关

代码<br>
[pretrained_xlnet](https://github.com/ymcui/Chinese-PreTrained-XLNet)<br>

