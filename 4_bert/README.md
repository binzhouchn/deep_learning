# BERT笔记

[**bert网址**](#bert网址)

[**使用BERT和TensorFlow构建文本分类器**](#使用bert和tensorflow构建文本分类器)

[**bert跑中文任务**](#bert跑中文任务)

[**bert DIY**](#bert_diy)

---

### bert网址

[github地址](https://github.com/google-research/bert)<br>

### 使用bert和tensorflow构建文本分类器

文本二分类任务，dataprocessor用cola<br>
```shell
python run_classifier.py --task_name=cola --do_train=true --do_eval=true --do_predict=true --data_dir=/home/zhoubin/bert/data/ --vocab_file=/home/zhoubin/bert/uncased_L-12_H-768_A-12/vocab.txt --bert_config_file=/home/zhoubin/bert/uncased_L-12_H-768_A-12/bert_config.json --init_checkpoint=/home/zhoubin/bert/uncased_L-12_H-768_A-12/bert_model.ckpt --max_seq_length=128 --train_batch_size=32 --learning_rate=2e-5 --num_train_epochs=5.0 --output_dir=/home/zhoubin/bert/output --do_lower_case=True
```

### bert跑中文任务




### bert_DIY

自定义Processor，继承DataProcessor<br>
修改input_fn_builder<br>
修改model_fn_builder和里面的create_model，其中create_model中output_layer后面可以自己接其他的层比如CRF层然后再改一下loss的计算方式或者不改<br>
bert create_model中最后一层其实就是Dense层和log_loss，我们可以直接用封装好的Dense不需要像源码自己写

