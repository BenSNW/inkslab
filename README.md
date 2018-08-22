# 深度自然语言处理库 inkslab

![inkslab](imgs/inkslab.jpg)

## 目录

- [新词发现](#1-新词发现)
- [关键词抽取](#2-关键字提取)
- [分词](#3-分词)
- [词性标注](#4-词性标注)
- [命名实体识别](#5-命名实体识别)
- [文本分类](#6-文本分类)
- [自动摘要](#7-自动摘要)
- [依存句法分析](#8-依存句法分析)


## 1 新词发现

```bash
cd inkslab/python/new_words
python test.py
```

更多详细信息见[新词发现](inkslab/python/new_words/README.md)

## 2 关键词抽取

```bash
cd inkslab/python/keywords
python test.py
```

更多详细信息见[关键词抽取](inkslab/python/keywords/README.md)

## 3 分词

## 3.1 训练模型

```bash
python inkslab/python/segments/main.py\
--process=train \
--workers=localhost:2220 \
--parameter_servers=localhost:2221 \
--job_name=ps/worker \
--task_index=0 \
--vocab_path=/Users/endy/nlp/inkslab/datas/segment/segment_vocab \
--word2vec_path=/Users/endy/nlp/inkslab/datas/common/char_vec.txt \
--train_data_path=/Users/endy/nlp/inkslab/datas/segment/train.txt \
--dev_data_path=/Users/endy/nlp/inkslab/datas/segment/dev.txt \
--test_data_path=/Users/endy/nlp/inkslab/datas/segment/test.txt \
--model_dir=/Users/endy/nlp/inkslab/models/segment \
--model_method=idcnn \
--train_steps=1500
```

## 3.2 导出模型

```bash
python tools/freeze_graph.py
--input_graph /Users/endy/nlp/inkslab/models/segment/graph.pbtxt \
--input_checkpoint /Users/endy/nlp/inkslab/models/segment/best_model \
--output_node_names "transitions,scores_1" \
--output_graph /Users/endy/nlp/inkslab/models/segment/seg_model.pbtxt
```

## 3.3 分词

```bash
python inkslab/python/segments/main.py\
--process=predict \
--vocab_path=/Users/endy/nlp/inkslab/datas/segment/segment_vocab \
--word2vec_path=/Users/endy/nlp/inkslab/datas/common/char_vec.txt \
--user_dict_path=/Users/endy/nlp/inkslab/datas/common/ud_dict.txt \
--raw_data_path=/Users/endy/nlp/inkslab/datas/segment/raw.txt \
--result_data_path=/Users/endy/nlp/inkslab/datas/segment/raw_seg.txt \
--model_dir=/Users/endy/nlp/inkslab/models/segment \ 
--model_name=seg_model.pbtxt
```

更多详细信息见[分词](inkslab/python/segments/README.md)

# 4 序列化标注

## 4.1 训练模型

```bash
python inkslab/python/pos_tagging/main.py\
--process=train \
--workers=localhost:2220 \
--parameter_servers=localhost:2221 \
--job_name=ps/worker \
--task_index=0 \
--vocab_path=/Users/endy/nlp/inkslab/datas/pos-tagging/tagging_vocab \
--word2vec_path=/Users/endy/nlp/inkslab/datas/common/char_vec.txt \
--train_data_path=/Users/endy/nlp/inkslab/datas/pos-tagging/train.txt \
--dev_data_path=/Users/endy/nlp/inkslab/datas/pos-tagging/dev.txt \
--test_data_path=/Users/endy/nlp/inkslab/datas/pos-tagging/test.txt \
--model_dir=/Users/endy/nlp/inkslab/models/pos-tagging \
--model_method=idcnn \
--train_steps=1500
```

## 4.2 导出模型

```bash
python tools/freeze_graph.py
--input_graph /Users/endy/nlp/inkslab/models/pos-tagging/graph.pbtxt \
--input_checkpoint /Users/endy/nlp/inkslab/models/pos-tagging/best_model \
--output_node_names "transitions,scores_1" \
--output_graph /Users/endy/nlp/inkslab/models/pos-tagging/tagging_model.pbtxt
```

## 4.3 词性预测

```bash
python inkslab/python/pos_tagging/main.py\
--process=predict \
--vocab_path=/Users/endy/nlp/inkslab/datas/pos-tagging/tagging_vocab \
--word2vec_path=/Users/endy/nlp/inkslab/datas/common/char_vec.txt \
--user_dict_path=/Users/endy/nlp/inkslab/datas/common/ud_dict.txt \
--raw_data_path=/Users/endy/nlp/inkslab/datas/pos-tagging/raw.txt \
--result_data_path=/Users/endy/nlp/inkslab/datas/pos-tagging/raw_pos.txt \
--model_dir=/Users/endy/nlp/inkslab/models/pos-tagging \ 
--model_name=tagging_model.pbtxt
```

更多详细信息见[词性标注](inkslab/python/pos_tagging/README.md)

# 5 命名实体识别

## 5.1 训练模型

```bash
python inkslab/python/pos_tagging/main.py\
--process=train \
--workers=localhost:2220,localhost:2221 \
--parameter_servers=localhost:2222 \
--job_name=ps/worker \
--task_index=0 \
--vocab_path=/Users/endy/nlp/inkslab/datas/ner/tagging_vocab \
--word2vec_path=/Users/endy/nlp/inkslab/datas/common/char_vec.txt \
--train_data_path=/Users/endy/nlp/inkslab/datas/ner/train.txt \
--dev_data_path=/Users/endy/nlp/inkslab/datas/ner/dev.txt \
--test_data_path=/Users/endy/nlp/inkslab/datas/ner/test.txt \
--model_dir=/Users/endy/nlp/inkslab/models/ner \
--model_method=idcnn \
--train_steps=1500
```

## 5.2 导出模型

```bash
python tools/freeze_graph.py
--input_graph /Users/endy/nlp/inkslab/models/ner/graph.pbtxt \
--input_checkpoint /Users/endy/nlp/inkslab/models/ner/best_model \
--output_node_names "transitions,scores_1" \
--output_graph /Users/endy/nlp/inkslab/models/ner/ner_model.pbtxt
```

## 5.3 命名实体识别

```bash
python inkslab/python/pos_tagging/main.py\
--process=predict \
--vocab_path=/Users/endy/nlp/inkslab/datas/ner/ner_vocab \
--word2vec_path=/Users/endy/nlp/inkslab/datas/common/char_vec.txt \
--user_dict_path=/Users/endy/nlp/inkslab/datas/common/ud_dict.txt \
--raw_data_path=/Users/endy/nlp/inkslab/datas/ner/raw.txt \
--result_data_path=/Users/endy/nlp/inkslab/datas/ner/raw_ner.txt \
--model_dir=/Users/endy/nlp/inkslab/models/ner \ 
--model_name=ner_model.pbtxt
```
更多详细信息见[命名实体识别](inkslab/python/ner/README.md)

# 6 文本分类

## 6.1 训练模型

```bash
python main.py
--process=train \
--workers=localhost:2220 \
--parameter_servers=localhost:2222 \
--job_name=worker \
--task_index=0 \
--log_dir=/Users/endy/nlp/inkslab/models/text-classification \
--train_path=/Users/endy/nlp/inkslab/datas/text-classification/train.tfrecord  \
--val_path=/Users/endy/nlp/inkslab/datas/text-classification/val.pkl \
--test_path=/Users/endy/nlp/inkslab/datas/text-classification/test.pkl \
--vocab_info_path=/Users/endy/nlp/inkslab/datas/text-classification/vocab_info.pkl \
--training_steps=1000 \
--model=RNN \
--learning_rate=0.1 \
--num_iter_decay=20
```

## 6.2 导出模型

```bash
python freeze_graph.py \
--input_graph=/Users/endy/nlp/inkslab/models/text-classification/best_model.pbtxt  \
--input_checkpoint=/Users/endy/nlp/inkslab/models/text-classification/best_model \
--output_graph=/Users/endy/nlp/inkslab/models/text-classification/RNN_frozen_model.pbtxt \
--input_binary=False \
--output_node_names=text_classification_1/loss,text_classification_1/prediction,text_classification_1/accuracy
```

## 6.3 预测

```bash
python main.py
--process=predict \
--log_dir=/Users/endy/nlp/inkslab/models/text-classification \
--test_path=/Users/endy/nlp/inkslab/datas/text-classification/test.pkl \
--vocab_info_path=/Users/endy/nlp/inkslab/datas/text-classification/vocab_info.pkl \
--model=RNN
```

# 7 自动摘要

## 7.1 训练模型

```bash
python inkslab/python/nmt_autosum/main.py \
--src=article \
--tgt=summary \
--vocab_prefix=/Users/endy/nlp/inkslab/datas/autosum/vocab \
--train_prefix=/Users/endy/nlp/inkslab/datas/autosum/train.10w \
--dev_prefix=/Users/endy/nlp/inkslab/datas/autosum/valid.1w \
--test_prefix=/Users/endy/nlp/inkslab/datas/autosum/valid.1w \
--out_dir=/Users/endy/nlp/inkslab/models/autosum/nmt_model \
--num_train_steps=50000 \
--steps_per_stats=200 \
--num_layers=2 \
--num_units=128 \
--dropout=0.2 \
--metrics=bleu \
--attention=bahdanau
```

## 7.2 生成摘要

```bash
python inkslab/python/nmt_autosum/main.py \
--inference_input_file=/Users/endy/nlp/inkslab/datas/autosum/test.txt \
--inference_output_file=/Users/endy/nlp/inkslab/datas/autosum/ouput.txt \
--out_dir=/Users/endy/nlp/inkslab/models/autosum/nmt_model
```

# 8 依存句法分析

## 8.1 训练模型

```bash
python inkslab/python/parse/main.py \
--process=train \
--workers=localhost:2220 \
--parameter_servers=localhost:2222 \ 
--job_name=ps/worker \
--task_index=0 \
--model_dir=/Users/endy/nlp/inkslab/models/parse \
--parse_data_path=/Users/endy/nlp/inkslab/datas/parse
```
## 8.2 导出模型

```bash
python inkslab/python/tools/freeze_graph.py
--input_graph=/Users/endy/nlp/inkslab/models/parse/graph.pbtxt
--input_checkpoint=/Users/endy/nlp/inkslab/models/parse/model.ckpt-0
--output_node_names="logit,loss"
--output_graph=/Users/endy/nlp/inkslab/models/parse/parse_model.pbtxt
```

## 8.3 预测

```bash
python inkslab/python/parse/main.py \
--process=predict \
--raw_data_path=/Users/endy/nlp/inkslab/datas/parse/raw.txt \
--result_data_path=/Users/endy/nlp/inkslab/datas/parse/raw_result.txt \
--model_dir=/Users/endy/nlp/inkslab/models/parse \
--parse_data_path=/Users/endy/nlp/inkslab/datas/parse 
```

