# 翻译或自动摘要

## 1 构建词典

```bash
python inkslab/python/dataset/handle_nmt_data.py \
--data_dir=/Users/endy/nlp/inkslab/datas/autosum
```

## 2 训练模型

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
--metrics=rouge \
--attention=bahdanau
```

## 3 生成摘要

```bash
python inkslab/python/nmt_autosum/main.py \
--inference_input_file=/Users/endy/nlp/inkslab/datas/autosum/test.txt \
--inference_output_file=/Users/endy/nlp/inkslab/datas/autosum/ouput.txt \
--out_dir=/Users/endy/nlp/inkslab/models/autosum/nmt_model
```