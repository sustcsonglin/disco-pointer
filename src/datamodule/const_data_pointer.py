
from supar.utils.common import *
from .dm_util.fields import SubwordField, Field, SpanField
import nltk
from fastNLP.core.dataset import DataSet
import logging
from .base import DataModuleBase
# from .dm_util.padder import SpanPadder, SpanLabelPadder
from .trees import load_trees,  tree2span, get_nongold_span
from src.datamodule.benepar.decode_chart import get_labeled_spans
from functools import cmp_to_key
from .trees import transition_system
from fastNLP.core.field import Padder
import numpy as np



log = logging.getLogger(__name__)

class ConstData4Pointer(DataModuleBase):
    def __init__(self, conf):
        super(ConstData4Pointer, self).__init__(conf)

    def get_inputs(self):
        return ['seq_len', 'chart',  'action_len', 'span_start', 'span_end', 'gold_span']

    def get_targets(self):
        return ['raw_tree', 'raw_word', 'raw_pos']

    def build_datasets(self):
        datasets = {}
        conf = self.conf
        datasets['train'] = self._load(const_file=conf.train_const)
        # if not self.conf.debug:
        datasets['dev'] = self._load(const_file=conf.dev_const)
        datasets['test'] = self._load(const_file=conf.test_const)
        return datasets

    def _load(self,  const_file):
        log.info(f'loading: {const_file}')
        dataset = DataSet()
        with open(const_file, encoding='utf-8') as f:
            raw_treebank = [line.rstrip() for line in f]
        trees, word, pos, raw_tree = get_pos_word_from_raw_tree(raw_treebank)
        spans = [get_labeled_spans(tree) for tree in trees]
        # length = [len(w) for w in word]
        # span_start = []
        # span_end = []
        # label = []
        # action_golds = []
        def compare(a, b):
            if a[0] > b[0]:
                return 1
            elif a[0] == b[0]:
                if a[1] > b[1]:
                    return -1
                else:
                    return 1
            else:
                return -1


        left = []
        right = []
        gold_spans = []
        labels = []
        for span in spans:
            span.sort(key=cmp_to_key(compare))
            ls = []
            rs = []
            g_span = []
            for s in span:
                l, r = s[:2]
                ls.append(l)
                rs.append(r)
                g_span.append([l,r])

            ls.append(0)
            rs.append(0)
            label =  [s[-1] for s in span]
            label.append('<end>')
            left.append(ls)
            right.append(rs)
            labels.append(label)
            gold_spans.append(g_span)




        dataset.add_field('span_start', left)
        dataset.add_field('gold_span', gold_spans, padder=SpanPadder(), is_input=True, ignore_type=True)
        dataset.add_field('span_end', right)
        dataset.add_field('chart', labels)
        dataset.add_field('word', word)
        dataset.add_field('pos', pos)
        dataset.add_field('raw_pos', pos, ignore_type=True, padder=None)
        dataset.add_field('action_len', [len(l) for l in left],)

        # dataset.add_field('chart', label)
        # dataset.add_field('span_start', span_start)
        # dataset.add_field('span_end', span_end)
        dataset.add_field('raw_tree', raw_tree, ignore_type=True, padder=None)
        #place holder
        dataset.add_field('char', word)
        dataset.add_field('raw_word', word, ignore_type=True, padder=None)
        dataset.add_field('raw_raw_word', word)
        dataset.add_seq_len("raw_word", 'seq_len')
        log.info(f'loading: {const_file} finished')
        return dataset

    def _set_padder(self, datasets):
        pass



    def build_fields(self, train_data):
        fields = {}
        fields['word'] = Field('word', pad=PAD, unk=UNK, bos=BOS, eos=EOS, lower=True, min_freq=self.conf.min_freq)
        fields['pos'] = Field('pos', pad=PAD, unk=UNK, bos=BOS, eos=EOS)
        fields['char'] = SubwordField('char', pad=PAD, unk=UNK, bos=BOS, eos=EOS, subword_eos=subword_eos, subword_bos=subword_bos,
                                      fix_len=self.conf.fix_len)
        fields['chart'] = Field('chart', pad=PAD, unk=UNK)

        for name, field in fields.items():
            field.build(train_data[name])
        return fields


def get_pos_word_from_raw_tree(raw_treebank):
    trees = []
    word = []
    pos = []
    tree_string = []
    for s in raw_treebank:
        if '(TOP' not in s:
            s = '(TOP ' + s + ')'
        tree = nltk.Tree.fromstring(s)
        w, p = zip(*tree.pos())
        word.append(w)
        pos.append(p)
        trees.append(tree)
        tree_string.append(s)
    return trees, word, pos, tree_string


class SpanPadder(Padder):
    def __call__(self, contents, field_name, field_ele_dtype, dim: int):
        # max_sent_length = max(rule.shape[0] for rule in contents)
        padded_array = []
        for b_idx, spans in enumerate(contents):
            for (start, end) in spans:
                padded_array.append([b_idx, start, end])
        return np.array(padded_array)


def process(spans, length, use_fine_grained=False):
    def compare(a, b):
        if a[0] > b[0]:
            return 1
        elif a[0] == b[0]:
            if a[1] > b[1]:
                return -1
            else:
                return 1
        else:
            return -1

    def compare2(a, b):
        if int(a[0]) >= int(b[1]):
            return 1

        elif b[0] <= a[0] <= a[1] <= b[1]:
            return -1

        elif a[0] <= b[0] <= b[1] <= a[1]:
            return 1

        elif b[0] >= a[1]:
            return -1

        else:
            raise ValueError
    sentence_len = length
    results = []
    spans.sort(key=cmp_to_key(compare))
    return spans
