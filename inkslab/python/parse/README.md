# 依存句法分析

## 1 解析conll数据

```bash
cd inkslab/python/dataset
python handle_conll_data.py --data_dir /Users/endy/nlp/inkslab/datas/parse
```

## 2 生成训练数据和验证数据

```bash
cd inkslab/python/parse/transition_system.py --data_dir /Users/endy/nlp/inkslab/datas/parse
```

## 3 训练模型

```bash
python inkslab/python/parse/main.py \
--process=train \
--workers=localhost:2220 \
--parameter_servers=localhost:2222 \ 
--job_name=ps/worker \
--task_index=0 \
--batch_size=10 \
--model_dir=/Users/endy/nlp/inkslab/models/parse \
--parse_data_path=/Users/endy/nlp/inkslab/datas/parse
```

## 4 导出模型

```bash
python inkslab/python/tools/freeze_graph.py
--input_graph=/Users/endy/nlp/inkslab/models/parse/graph.pbtxt
--input_checkpoint=/Users/endy/nlp/inkslab/models/parse/model.ckpt-0
--output_node_names="logit,loss"
--output_graph=/Users/endy/nlp/inkslab/models/parse/parse_model.pbtxt
```

## 4 预测

```bash
python inkslab/python/parse/main.py \
--process=predict \
--batch_size=1 \
--raw_data_path=/Users/endy/nlp/inkslab/datas/parse/raw.txt \
--result_data_path=/Users/endy/nlp/inkslab/datas/parse/raw_result.txt \
--model_dir=/Users/endy/nlp/inkslab/models/parse \
--parse_data_path=/Users/endy/nlp/inkslab/datas/parse
```