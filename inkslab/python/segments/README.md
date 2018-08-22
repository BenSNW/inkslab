# 分词

## 1 创建词典并生成数据集

```bash
python inkslab/python/dataset/handle_people_daily.py \
--data_dir /Users/endy/nlp/inkslab/datas/common/2014
--vocab_file /Users/endy/nlp/inkslab/datas/segment/segment_vocab
--tag_file /Users/endy/nlp/inkslab/datas/segment/tags.txt
--data_output_file /Users/endy/nlp/inkslab/datas/segment/full_id.txt
--max_sentence_len 100
--task_type segment
```

## 2 生成训练集、验证集以及测试集

```bash
python inkslab/python/dataset/split_dataset.py \
/Users/endy/nlp/inkslab/datas/segment/full_id.txt \
/Users/endy/nlp/inkslab/datas/segment/train.txt \
/Users/endy/nlp/inkslab/datas/segment/dev.txt \
/Users/endy/nlp/inkslab/datas/segment/test.txt
```

## 3 训练模型

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

## 4 导出模型

```bash
python tools/freeze_graph.py
--input_graph /Users/endy/nlp/inkslab/models/segment/graph.pbtxt \
--input_checkpoint /Users/endy/nlp/inkslab/models/segment/best_model \
--output_node_names "transitions,scores_1" \
--output_graph /Users/endy/nlp/inkslab/models/segment/seg_model.pbtxt
```

## 5 分词

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

## 6 参考文献

- [Neural Architectures for Named Entity Recognition](http://www.aclweb.org/anthology/N16-1030)
- [Fast and Accurate Entity Recognition with Iterated Dilated Convolutions](https://arxiv.org/abs/1702.02098)