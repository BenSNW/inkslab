# -*- coding:utf-8 -*-

import os
import numpy as np
from inkslab.python.dataset.dataset_utils import load_pickle_data
from inkslab.python.tf_model import TfModel


def predictor(args):
    id2word, word2id, id2label, label2id, word2vec \
        = load_pickle_data(args.vocab_info_path)

    output_graph_path = os.path.join(args.log_dir, args.model+"_frozen_model.pbtxt")
    print(output_graph_path)

    model = TfModel()
    load_suc = model.load(output_graph_path)
    if not load_suc:
        print("Could not load model from: " + output_graph_path)
        return False

    tx, ty = load_pickle_data(args.test_path)    # 直接读取测试数据集
    tx = np.array(tx)
    ty = np.array(ty)

    batch_size = args.batch_size
    total_len = tx.shape[0]
    num_batch = int((tx.shape[0] - 1) / batch_size) + 1
    batch_size_list = []
    accuracy_list = []
    prediction_list = []
    loss_list = []

    loss = model.get_tensor('text_classification_1/loss:0')
    prediction = model.get_tensor('text_classification_1/prediction:0')
    accuracy = model.get_tensor('text_classification_1/accuracy:0')

    input_x = model.get_tensor('tx_in:0')
    input_y = model.get_tensor('ty_in:0')

    for i in range(num_batch):
        end_off = (i + 1) * batch_size
        if end_off > total_len:
            end_off = total_len
        sample_num = end_off - i * batch_size
        curr_x = tx[i * batch_size: end_off]
        curr_y = ty[i * batch_size: end_off]
        loss_value, prediction_value, accuracy_value =\
            model.run([loss, prediction, accuracy], {input_x: curr_x, input_y: curr_y})

        batch_size_list.append(sample_num)
        accuracy_list.append(accuracy_value * sample_num)

        for v in prediction_value:
            prediction_list.append(id2label[v])
        loss_list.append(loss_value * sample_num)

    accuracy_value = sum(accuracy_list) / max(sum(batch_size_list), 1)
    loss_value = sum(loss_list) / max(sum(batch_size_list), 1)
    print("Accuracy: %.3f, loss: %.3f" % (accuracy_value, loss_value))
    return prediction_list, accuracy_value, loss_value