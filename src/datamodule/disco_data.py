
from supar.utils.common import *
from .dm_util.fields import SubwordField, Field, SpanField
import nltk
from fastNLP.core.dataset import DataSet
import logging
from .base import DataModuleBase
from .dm_util.padder import SpanPadder, SpanLabelPadder
from .disco_tree import Token, Tree, get_yield
import pdb
import itertools
from functools import cmp_to_key


def nltk_tree_to_Tree(nltk_tree):
    # Leaf
    if len(nltk_tree) == 1 and type(nltk_tree[0]) == str:
        idx, token = nltk_tree[0].split("=", 1)
        idx = int(idx)
        return Token(token, idx, [nltk_tree.label()])
    else:
        children = [nltk_tree_to_Tree(child) for child in nltk_tree]
        return Tree(nltk_tree.label(), children)

def read_discbracket_corpus(filename):
    from nltk import Tree as nTree
    with open(filename) as f:
        ctrees = [nTree.fromstring(line.strip()) for line in f]

    result = [nltk_tree_to_Tree(t) for t in ctrees]
    return result



log = logging.getLogger(__name__)

from functools import cmp_to_key

class DiscoData(DataModuleBase):
    def __init__(self, conf):
        super(DiscoData, self).__init__(conf)

    def get_inputs(self):
        return ['seq_len', 'chart',  'action_len', 'span_start', 'span_end', 'gold_c', 'gold_d']


    def get_targets(self):
        return ['raw_tree', 'raw_word', 'raw_pos', 'id']


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
        treebank = read_discbracket_corpus(const_file)

        with open(const_file) as f:
            raw_treebank = [line.rstrip() for line in f]

        words, poses, raw_pos = [], [], []
        left = []
        right = []
        labels = []

        constituents= []
        discontituents = []
        def compare(a, b):
            if len(a) == 3:
                last_a =  a[1]
                len_a = a[1] - a[0]
            else:
                last_a = a[0][-1][-1]
                len_a = 0
                for span in a[0]:
                    len_a += (span[1] - span[0])

            if len(b) == 3:
                last_b =  b[1]
                len_b = b[1] - b[0]
            else:
                last_b = b[0][-1][-1]
                len_b = 0
                for span in b[0]:
                    len_b += (span[1] - span[0])


            if last_b < last_a:
                return 1
            elif last_b > last_a:
                return -1
            else:
                if len_b > len_a:
                    return -1
                elif len_b < len_a:
                    return 1
                else:
                    return 0

        cmp = cmp_to_key(compare)


        for tree in treebank:
            tree.merge_unaries()
            word, pos = tree.get_words()
            words.append(word)
            raw_pos.append(pos)
            pos = [p[0] for p in pos]
            poses.append(pos)
            disco = tree.dis()
            co = tree.cont()

            span = co + disco
            span.sort(key=cmp)

            new_span = []
            for s in span:
                if len(s) == 3:
                    new_span.append(s)
                else:
                    subspans, l =  s
                    for i in range(len(subspans) - 1):
                        new_span.append( [subspans[i][0], subspans[i][1], "<disco>" ])
                    new_span.append([subspans[-1][0], subspans[-1][1], l])

            span_l = [n[0] for n in new_span]
            span_r = [n[1] for n in new_span]
            span_label = [n[2] for n in new_span]
            span_l.append(0)
            span_r.append(0)
            span_label.append('<end>')
            left.append(span_l)
            right.append(span_r)
            labels.append(span_label)
            constituents.append(co)
            discontituents.append(disco)

        dataset.add_field('span_start', left)
        dataset.add_field('span_end', right)
        dataset.add_field('chart', labels)
        dataset.add_field('action_len', [len(l) for l in left],)

        dataset.add_field('gold_c', constituents, ignore_type=True, padder=None)
        # dataset.add_field('valid', is_valid)
        dataset.add_field('gold_d', discontituents, ignore_type=True, padder=None)
        dataset.add_field('id', [i for i in range(len(dataset))])

        #place holder
        dataset.add_field('char', words)
        dataset.add_field('word', words)
        dataset.add_field('pos', poses)
        dataset.add_field('raw_pos', raw_pos, ignore_type=True)
        dataset.add_field('raw_word', words)
        dataset.add_field('raw_raw_word', words)
        dataset.add_field('id', [i for i in range(len(dataset))])

        # seq_len & action_len
        dataset.add_seq_len("raw_word", 'seq_len')
        # dataset.add_seq_len("cursor", 'action_len')
        # for eval.
        dataset.add_field('raw_tree', raw_treebank, padder=None, ignore_type=True)
        log.info(f'loading: {const_file} finished')
        return dataset

    # def _set_padder(self, datasets):
    #     for _, dataset in datasets.items():
    #         dataset.add_field('chart', dataset['chart'].content, padder=CoSpanLabelPadder(), ignore_type=True, is_input=True)
    #         dataset.add_field('chart_d', dataset['chart_d'].content, padder=DiscoSpanLabelPadder(), ignore_type=True, is_input=True)

    def build_fields(self, train_data):
        fields = {}
        fields['word'] = Field('word', pad=PAD, unk=UNK, bos=BOS, eos=EOS, lower=True, min_freq=self.conf.min_freq)
        fields['pos'] = Field('pos', pad=PAD, unk=UNK, bos=BOS, eos=EOS)
        fields['char'] = SubwordField('char', pad=PAD, unk=UNK, bos=BOS, eos=EOS, subword_eos=subword_eos, subword_bos=subword_bos,
                                      fix_len=self.conf.fix_len)
        fields['chart'] = Field('chart', pad=PAD, unk=UNK)
        for name, field in fields.items():
            field.build(train_data[name])
            # pdb.set_trace()
        # pdb.set_trace()
        return fields


