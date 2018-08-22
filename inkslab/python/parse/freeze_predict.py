# -*- coding: utf-8 -*-

import os
import numpy as np
from inkslab.python.tf_model import TfModel
from inkslab.python.dataset import handle_conll_data
from inkslab.python.parse import transition_system
from inkslab.python.dataset.handle_conll_data import Sentence
from inkslab.python.parse.transition_system import Configuration
from inkslab.python.dataset.handle_conll_data import DependencyTree
from inkslab.python.dataset.constants import UNKNOWN


class Predictor(object):
    def __init__(self, args):
        self.data_path = args.parse_data_path
        self.model_dir = args.model_dir
        self.output_graph_path = os.path.join(args.model_dir, "parse_model.pbtxt")
        self.model = TfModel()
        self.load_model()
        # data utils for parsing:
        vocab_dict_path = os.path.join(self.data_path, "vocab_dict")
        pos_dict_path = os.path.join(self.data_path, "pos_dict")
        label_dict_path = os.path.join(self.data_path, "label_dict")
        feature_tpl_path = os.path.join(self.data_path, "parse.template")

        self.vocab_dict = handle_conll_data.read_vocab(vocab_dict_path)
        self.pos_dict = handle_conll_data.read_vocab(pos_dict_path)
        self.label_dict = handle_conll_data.read_vocab(label_dict_path)
        self.feature_tpl = handle_conll_data.read_template(feature_tpl_path)
        self.arc_labels = transition_system.generate_arcs(self.label_dict)  # Arc Labels: <K,V>   (L(1), 1)

        self.rev_vocab_dict = handle_conll_data.reverse_map(self.vocab_dict)
        self.rev_pos_dict = handle_conll_data.reverse_map(self.pos_dict)
        self.rev_label_dict = handle_conll_data.reverse_map(self.label_dict)
        self.rev_arc_labels = handle_conll_data.reverse_map(self.arc_labels)

        print("NOTICE: vocab_dict size %d" % len(self.vocab_dict))
        print("NOTICE: pos_dict size %d" % len(self.pos_dict))
        print("NOTICE: label_dict size %d" % len(self.label_dict))
        print("NOTICE: arc_labels size %d" % len(self.arc_labels))
        print("NOTICE: feature templates size %d" % len(self.feature_tpl))

    def load_model(self):
        load_suc = self.model.load(self.output_graph_path)
        if not load_suc:
            print("Could not load model from: " + self.output_graph_path)
            return False
        else:
            return True

    def predict(self, words, tags):
        dep_tree = self._predict(words, tags)
        return dep_tree

    def _predict(self, words, tags):
        if len(words) != len(tags):
            print("words list and tags list length is different")
            return
        if len(words) == 0 or len(tags) == 0:
            print("DEBUG: words or tags list is empty")
            return
        # convert words and tags to Sentence object, sentence save the word_id and tag_id; UNKNOWN is '*
        sent = Sentence()
        for w, t in zip(words, tags):
            w_id = self.vocab_dict[w] if w in self.vocab_dict.keys() else self.vocab_dict[UNKNOWN]
            t_id = self.pos_dict[t] if t in self.pos_dict.keys() else self.pos_dict[UNKNOWN]
            sent.add(w_id, t_id)

        tree_idx = self._predict_tree(sent)

        # New Dependency Tree object to convert label ids to actual labels, and words, pos tags, etc.
        num_token = tree_idx.count()
        tree_label = DependencyTree()  # final tree to store the label value of dependency
        for i in range(0, num_token):
            cur_head = tree_idx.get_head(i + 1)
            cur_label_id = tree_idx.get_label(i + 1)  # 0 is ROOT, actual index start from 1
            cur_label_name = self.rev_label_dict[cur_label_id] if cur_label_id in self.rev_label_dict.keys() else ""
            tree_label.add(words[i], tags[i], cur_head, cur_label_name)
        return tree_label

    def _predict_tree(self, sent):
        target_num = len(self.rev_arc_labels)
        # print ("target number is %d" % target_num)
        config = Configuration(sent)

        while not config.is_terminal():
            features = transition_system.get_features(config, self.feature_tpl)  # 1-D list
            X = np.array([features])  # convert 1-D list to ndarray size [1, dim]
            Y = np.array([[0] * target_num])

            input_x = self.model.get_tensor('tx_in:0')
            input_y = self.model.get_tensor('ty_in:0')

            logits_tensor = self.model.get_tensor('logit:0')

            logit = self.model.run([logits_tensor], {input_x: X, input_y: Y})

            pred_next_arc_id = int(np.argmax(logit))  # prediction of next arc_idx of 1 of (2 * Nl +1)
            pred_next_arc = self.rev_arc_labels[pred_next_arc_id]  # 5-> L(2)   idx -> L(label_id)
            config.step(pred_next_arc)  # Configuration Take One Step
            # When config is terminal, return the final dependence trees object

        dep_tree = config.tree
        return dep_tree


def predict(args):
    parser = Predictor(args)
    with open(args.raw_data_path, 'r') as reader, open(args.result_data_path, 'w') as writer:

        writer.writelines("id\tword\tpos\thead\tlabel\n")
        for line in reader.readlines():
            terms = line.split()
            words = [term.split('/')[0] for term in terms]
            tags = [term.split('/')[1] for term in terms]
            # Parsing
            dep_tree = parser.predict(words, tags)
            num_token = dep_tree.count()
            writer.writelines("\n")
            for i in range(num_token):
                cur_id = str(dep_tree.tree[i + 1].id)
                cur_form = str(dep_tree.tree[i + 1].form)
                cur_pos = str(dep_tree.tree[i + 1].pos)
                cur_head = str(dep_tree.tree[i + 1].head)
                cur_label = str(dep_tree.tree[i + 1].deprel)
                writer.writelines(cur_id + "\t")
                writer.writelines(cur_form + "\t")
                writer.writelines(cur_pos + "\t")
                writer.writelines(cur_head + "\t")
                writer.writelines(cur_label + "\t")
                writer.writelines("\n")
