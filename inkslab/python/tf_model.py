# -*- coding: utf-8 -*-

import tensorflow as tf


class TfModel(object):
    def __init__(self):
        self._session_ = None
        self._graph = None

    def load(self, model_file):
        self._graph = tf.Graph()
        with self._graph.as_default():
            output_graph_def = tf.GraphDef()
            with open(model_file, "rb") as f:
                output_graph_def.ParseFromString(f.read())
                _ = tf.import_graph_def(output_graph_def, name="")
            self._session_ = tf.Session()
        return True

    def eval(self, name):
        tran_tensor = self._graph.get_tensor_by_name(name)
        results = self._session_.run(tran_tensor)
        return results

    def eval_with_input(self, inputs, name):
        input_data = self._graph.get_tensor_by_name('input_placeholder:0')
        tran_tensor = self._graph.get_tensor_by_name(name)
        results = self._session_.run(tran_tensor, feed_dict={input_data: inputs})
        return results

    def get_tensor(self, name):
        return self._graph.get_tensor_by_name(name)

    def run(self, inputs, feeds):
        results = self._session_.run(inputs, feeds)
        return results
