# 文本分类

## 1 数据预处理

```bash
python handle_sogou_data.py
--train_dir /Users/endy/nlp/inkslab/datas/text-classification \
--train_data_path /Users/endy/nlp/inkslab/datas/text-classification/sogou_title_label_train.csv \
--val_data_path /Users/endy/nlp/inkslab/datas/text-classification/sogou_title_label_val.csv \
--test_data_path /Users/endy/nlp/inkslab/datas/text-classification/sogou_title_label_test.csv \
--min_count 3 \
--pre_trained_vec_path /Users/endy/nlp/inkslab/datas/common/char_vec.txt \
--emb_dim 100 \
--max_sentence_length 20 \
--min_sentence_length 4
```

在数据预处理当中，如果选择加载与训练的模型，那么处理之后的词表信息会保存在'vocab_info_pre_trained_vec.pkl'当中，
否则会保存在'vocab_info.pkl'当中。

## 2 训练模型

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

## 3 导出模型

```bash
python freeze_graph.py \
--input_graph=/Users/endy/nlp/inkslab/models/text-classification/best_model.pbtxt  \
--input_checkpoint=/Users/endy/nlp/inkslab/models/text-classification/best_model \
--output_graph=/Users/endy/nlp/inkslab/models/text-classification/RNN_frozen_model.pbtxt \
--input_binary=False \
--output_node_names=text_classification_1/loss,text_classification_1/prediction,text_classification_1/accuracy
```

## 4 预测

```bash
python main.py
--process=predict \
--log_dir=/Users/endy/nlp/inkslab/models/text-classification \
--test_path=/Users/endy/nlp/inkslab/datas/text-classification/test.pkl \
--vocab_info_path=/Users/endy/nlp/inkslab/datas/text-classification/vocab_info.pkl \
--model=RNN
```