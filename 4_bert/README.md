# BERT笔记

[**bert网址**](#bert网址)

[**使用BERT和TensorFlow构建文本分类器**](#使用bert和tensorflow构建文本分类器)


---

### bert网址

[github地址](https://github.com/google-research/bert)<br>

### 使用bert和tensorflow构建文本分类器

文本二分类任务，dataprocessor用cola<br>
```shell
python run_classifier.py --task_name=cola --do_train=true --do_eval=true --do_predict=true --data_dir=/home/zhoubin/bert/data/ --vocab_file=/home/zhoubin/bert/uncased_L-12_H-768_A-12/vocab.txt --bert_config_file=/home/zhoubin/bert/uncased_L-12_H-768_A-12/bert_config.json --init_checkpoint=/home/zhoubin/bert/uncased_L-12_H-768_A-12/bert_model.ckpt --max_seq_length=128 --train_batch_size=32 --learning_rate=2e-5 --num_train_epochs=5.0 --output_dir=/home/zhoubin/bert/output --do_lower_case=True
```

