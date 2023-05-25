
import logging
import pdb
import sys
from asyncio import Queue
from collections import Counter
from queue import Empty

import nltk
import subprocess
import torch
from pytorch_lightning.metrics import Metric
from threading import Thread
import regex

from supar.utils.transform import Tree
import tempfile

log = logging.getLogger(__name__)
import os

from pathlib import Path
from functools import cmp_to_key


class SpanMetric(Metric):
    DELETE = {'TOP', 'S1', '-NONE-', ',', ':', '``', "''", '.', '?', '!', ''}
    EQUAL = {'ADVP': 'PRT'}

    def __init__(self, cfg, fields):
        super().__init__()
        self.cfg = cfg
        self.fields = fields
        self.vocab = fields.get_vocab('chart')
        self.add_state("n", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("n_ucm", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("n_multiple", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("c_multiple", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("n_lcm", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("utp", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("ltp", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("gold", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("pred", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.eps = 1e-12
        self.write_result_to_file = cfg.write_result_to_file
        # why i write this?
        self.transform = True
        if self.cfg.write_result_to_file:
            self.add_state("outputs", default=[])
            self.prefix = "const"



    def __call__(self, preds, golds):
        self.update(preds, golds)
        return self



    def update(self, info):

        preds = info['chart_preds']
        raw_preds = [str(tree) for tree in preds]
        golds = info['raw_tree']
        golds = [nltk.Tree.fromstring(tree) for tree in golds]
        preds = [nltk.Tree.fromstring(str(tree)) for tree in preds]
        preds = [Tree.factorize(tree, self.DELETE, self.EQUAL) for tree in preds]
        trees = [Tree.factorize(tree, self.DELETE, self.EQUAL) for tree in golds]
        try:
            assert len(preds) == len(trees)
        except:
            pdb.set_trace()
        _n_ucm, _n_lcm, _utp, _ltp, _pred, _gold = 0, 0, 0, 0, 0, 0
        self.n += len(preds)
        _n_multiple = 0
        _c_multiple = 0

        if self.cfg.write_result_to_file:
            output = {}
            output['raw_tree'] =  raw_preds
            output['gold_spans'] = trees
            output['pred_spans'] = preds
            output['id'] = info['word_id']
            self.outputs.append(output)

        for pred, gold in zip(preds, trees):
            upred = Counter([(i, j) for i, j, _ in pred])
            ugold = Counter([(i, j) for i, j, _ in gold])
            utp = list((upred & ugold).elements())
            lpred = Counter(pred)
            lgold = Counter(gold)
            ltp = list((lpred & lgold).elements())
            _n_ucm += float(len(utp) == len(pred) == len(gold))
            _n_lcm += float(len(ltp) == len(pred) == len(gold))
            _utp += len(utp)
            _ltp += len(ltp)
            _pred += len(pred)
            _gold += len(gold)

        self.n_ucm += _n_ucm
        self.n_lcm += _n_lcm
        self.utp += _utp
        self.ltp += _ltp
        self.pred += _pred
        self.gold += _gold

    def compute(self, test=True, epoch_num=-1):
        super(SpanMetric, self).compute()
        # if self.cfg.write_result_to_file and epoch_num > 0:
        if self.cfg.write_result_to_file:
            self._write_result_to_file(test=test)
        return self.result

    @property
    def result(self):
        return {
            'c_ucm': self.ucm(),
            'c_lcm': self.lcm(),
            'up': self.up(),
            'ur': self.ur(),
            'uf': self.uf(),
            'lp': self.lp(),
            'lr': self.lr(),
            'lf': self.lf(),
            'score': self.lf(),
        }

    def score(self):
        return self.lf()

    def ucm(self):
        return ( self.n_ucm / (self.n + self.eps)).item()

    def lcm(self):
        return (self.n_lcm / (self.n + self.eps)).item()

    def up(self):
        return (self.utp / (self.pred + self.eps)).item()

    def ur(self):
        return (self.utp / (self.gold + self.eps)).item()

    def uf(self):
        return (2 * self.utp / (self.pred + self.gold + self.eps)).item()

    def lp(self):
        return (self.ltp / (self.pred + self.eps)).item()

    def lr(self):
        return (self.ltp / (self.gold + self.eps)).item()

    def lf(self):
        return (2 * self.ltp / (self.pred + self.gold + self.eps)).item()

    def _write_result_to_file(self, test=False):
        mode = 'test' if test else 'valid'
        outputs = self.outputs


        ids = [output['id'] for output in outputs]
        raw_tree = [output['raw_tree'] for output in outputs]
        pred_spans = [output['pred_spans'] for output in outputs]
        gold_spans = [output['gold_spans'] for output in outputs]

        total_len =  sum(batch.shape[0] for batch in ids)

        final_results = [None for _ in range(total_len)]

        for batch in zip(ids, raw_tree, pred_spans, gold_spans):
            batch_ids, batch_raw_tree, batch_pred_span, batch_gold_span = batch
            for i in range(batch_ids.shape[0]):
                # length = len(batch_word[i])
                # recall that the first token is the imaginary root;
                a = []
                a.append(batch_raw_tree[i])
                a.append(batch_pred_span[i])
                a.append(batch_gold_span[i])

                final_results[batch_ids[i]] = a


        with open(f"{self.prefix}_output_{mode}.txt", 'w', encoding='utf8') as f:
            for (raw_tree, pred_span, gold_span) in final_results:
                f.write(raw_tree)
                f.write('\n')
                # f.write(f'pred_spans:{pred_span}')
                # f.write('\n')
                # f.write(f'gold_spans:{gold_span}')
                # f.write('\n')



class SpanMetric_SPMRL(Metric):
    DELETE = {'TOP', 'ROOT', 'S1', '-NONE-', 'VROOT'}
    EQUAL = {}

    def __init__(self, cfg, fields):
        super().__init__()
        self.cfg = cfg
        self.fields = fields
        self.vocab = fields.get_vocab('chart')
        self.add_state("n", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("n_ucm", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("n_multiple", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("c_multiple", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("n_lcm", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("utp", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("ltp", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("gold", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("pred", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.eps = 1e-12
        self.write_result_to_file = cfg.write_result_to_file
        # why i write this?
        self.transform = True
        if self.cfg.write_result_to_file:
            self.add_state("outputs", default=[])
            self.prefix = "const"



    def __call__(self, preds, golds):
        self.update(preds, golds)
        return self


    def build(self, tree, span):

        leaves = [subtree for subtree in tree.subtrees()
                  if not isinstance(subtree[0], nltk.Tree)]

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

        span.sort(key=cmp_to_key(compare))
        idx = -1

        def helper():
            nonlocal idx
            idx += 1
            i, j, label = span[idx]
            if (i + 1) >= j:
                children = [leaves[i]]
            else:
                children = []
                while (
                        (idx + 1) < len(span)
                        and i <= span[idx + 1][0]
                        and span[idx + 1][1] <= j
                ):
                    children.extend(helper())

            if label:
                if label == 'NULL' or ('<>' in label) or (label == '<pad>'):
                    return children

                for sublabel in reversed(label.split("+")):
                    children = [nltk.Tree(sublabel, children)]

            return children

        children = helper()
        new =  nltk.Tree("TOP", children)
        # try:
        assert len(new.pos()) == len(tree.pos())

        return new

    def update(self, info):

        preds = info['chart_preds']
        raw_preds = [str(tree) for tree in preds]
        golds = info['raw_tree']
        golds = [nltk.Tree.fromstring(tree) for tree in golds]
        preds = [nltk.Tree.fromstring(str(tree)) for tree in preds]
        # preds = [ [[i, j, self.vocab[k] if k != -1 else 'NULL'] for i,j,k in tree] for tree in preds]
        # preds = [
        #     self.build(tree, chart)
        #     for tree, chart in zip(golds, preds)
        # ]
        preds = [Tree.factorize(tree, self.DELETE, self.EQUAL) for tree in preds]
        trees = [Tree.factorize(tree, self.DELETE, self.EQUAL) for tree in golds]
        try:
            assert len(preds) == len(trees)
        except:
            pdb.set_trace()
        _n_ucm, _n_lcm, _utp, _ltp, _pred, _gold = 0, 0, 0, 0, 0, 0

        self.n += len(preds)

        _n_multiple = 0
        _c_multiple = 0

        if self.cfg.write_result_to_file:
            output = {}
            output['raw_tree'] =  raw_preds
            output['gold_spans'] = trees
            output['pred_spans'] = preds
            output['id'] = info['word_id']
            self.outputs.append(output)

        for pred, gold in zip(preds, trees):
            upred = Counter([(i, j) for i, j, _ in pred])
            ugold = Counter([(i, j) for i, j, _ in gold])
            utp = list((upred & ugold).elements())
            lpred = Counter(pred)
            lgold = Counter(gold)
            ltp = list((lpred & lgold).elements())
            _n_ucm += float(len(utp) == len(pred) == len(gold))
            _n_lcm += float(len(ltp) == len(pred) == len(gold))
            _utp += len(utp)
            _ltp += len(ltp)
            _pred += len(pred)
            _gold += len(gold)

        self.n_ucm += _n_ucm
        self.n_lcm += _n_lcm
        self.utp += _utp
        self.ltp += _ltp
        self.pred += _pred
        self.gold += _gold

    def compute(self, test=True, epoch_num=-1):
        super(SpanMetric_SPMRL, self).compute()
        # if self.cfg.write_result_to_file and epoch_num > 0:
        if self.cfg.write_result_to_file:
            self._write_result_to_file(test=test)
        return self.result


    @property
    def result(self):
        return {
            'c_ucm': self.ucm(),
            'c_lcm': self.lcm(),
            'up': self.up(),
            'ur': self.ur(),
            'uf': self.uf(),
            'lp': self.lp(),
            'lr': self.lr(),
            'lf': self.lf(),
            'score': self.lf(),
        }

    def score(self):
        return self.lf()

    def ucm(self):
        return ( self.n_ucm / (self.n + self.eps)).item()

    def lcm(self):
        return (self.n_lcm / (self.n + self.eps)).item()

    def up(self):
        return (self.utp / (self.pred + self.eps)).item()

    def ur(self):
        return (self.utp / (self.gold + self.eps)).item()

    def uf(self):
        return (2 * self.utp / (self.pred + self.gold + self.eps)).item()

    def lp(self):
        return (self.ltp / (self.pred + self.eps)).item()

    def lr(self):
        return (self.ltp / (self.gold + self.eps)).item()

    def lf(self):
        return (2 * self.ltp / (self.pred + self.gold + self.eps)).item()

    def _write_result_to_file(self, test=False):
        mode = 'test' if test else 'valid'
        outputs = self.outputs


        ids = [output['id'] for output in outputs]
        raw_tree = [output['raw_tree'] for output in outputs]
        pred_spans = [output['pred_spans'] for output in outputs]
        gold_spans = [output['gold_spans'] for output in outputs]

        total_len =  sum(batch.shape[0] for batch in ids)

        final_results = [None for _ in range(total_len)]

        for batch in zip(ids, raw_tree, pred_spans, gold_spans):
            batch_ids, batch_raw_tree, batch_pred_span, batch_gold_span = batch
            for i in range(batch_ids.shape[0]):
                # length = len(batch_word[i])
                # recall that the first token is the imaginary root;
                a = []
                a.append(batch_raw_tree[i])
                a.append(batch_pred_span[i])
                a.append(batch_gold_span[i])

                final_results[batch_ids[i]] = a


        with open(f"{self.prefix}_output_{mode}.txt", 'w', encoding='utf8') as f:
            for (raw_tree, pred_span, gold_span) in final_results:
                f.write(raw_tree)
                f.write('\n')
                # f.write(f'pred_spans:{pred_span}')
                # f.write('\n')
                # f.write(f'gold_spans:{gold_span}')
                # f.write('\n')




class DiscoSpanMetric(Metric):
    def __init__(self, cfg, fields):
        super().__init__()

        self.fields = fields
        self.cfg = cfg
        self.add_state("outputs", default=[])
        self.add_state("n", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("f", default=torch.tensor(0.), dist_reduce_fx='sum')
        self.add_state("f_", default=torch.tensor(0.), dist_reduce_fx='sum')
        self.add_state("tp", default=torch.tensor(0.), dist_reduce_fx='sum')
        self.add_state("fp", default=torch.tensor(0.), dist_reduce_fx='sum')
        self.add_state("fn", default=torch.tensor(0.), dist_reduce_fx='sum')

        self.add_state("tp_d", default=torch.tensor(0.), dist_reduce_fx='sum')
        self.add_state("fp_d", default=torch.tensor(0.), dist_reduce_fx='sum')
        self.add_state("fn_d", default=torch.tensor(0.), dist_reduce_fx='sum')

        self.add_state("p", default=torch.tensor(0.), dist_reduce_fx='sum')
        self.add_state("p_", default=torch.tensor(0.), dist_reduce_fx='sum')
        self.add_state("r", default=torch.tensor(0.), dist_reduce_fx='sum')
        self.add_state("r_", default=torch.tensor(0.), dist_reduce_fx='sum')
        self.add_state("d_f", default=torch.tensor(0.), dist_reduce_fx='sum')
        self.add_state("d_f_", default=torch.tensor(0.), dist_reduce_fx='sum')
        self.add_state("d_p", default=torch.tensor(0.), dist_reduce_fx='sum')
        self.add_state("d_p_", default=torch.tensor(0.), dist_reduce_fx='sum')
        self.add_state("d_r", default=torch.tensor(0.), dist_reduce_fx='sum')
        self.add_state("d_r_", default=torch.tensor(0.), dist_reduce_fx='sum')
        self.prefix = "ud"
        # self.ignore_punct = True

    def update(self, ctx):
        pred_tree = ctx['pred_tree']
        batch_id = ctx['id']
        outputs = {}
        outputs['pred_tree'] = pred_tree
        outputs['id'] = batch_id
        self.outputs.append(outputs)

        pred_cs = ctx['pred_c']
        pred_ds = ctx['pred_d']
        gold_cs = ctx['gold_c']
        gold_ds = ctx['gold_d']

        for i in range(batch_id.shape[0]):
            pred_c = pred_cs[i]
            pred_d = pred_ds[i]
            gold_c = gold_cs[i]
            gold_d = gold_ds[i]

            for span in pred_c:
                if span in gold_c:
                    self.tp += 1
                else:
                    self.fp += 1

            for span_d in pred_d:
                # pdb.set_trace()
                if span_d in gold_d:
                    self.tp += 1
                    self.tp_d += 1
                else:
                    self.fp += 1
                    self.fp_d += 1

            for span in gold_c:
                if span not in pred_c:
                    self.fn += 1


            for span_d in gold_d:
                if span_d not in pred_d:
                    self.fn_d += 1
                    self.fn += 1

    @property
    def result(self):
        prec = self.tp / (self.tp + self.fp + 1e-9)
        recall = self.tp / (self.tp + self.fn + 1e-9)
        f =   2 * prec * recall / (prec + recall) if (prec + recall > 0) else 0

        prec_d = self.tp_d / (self.tp_d + self.fp_d + 1e-9)
        recall_d = self.tp_d / (self.tp_d + self.fn_d + 1e-9)
        f_d =   2 * prec_d * recall_d / (prec_d + recall_d) if (prec_d + recall_d > 0) else 0

        return {'score': self.f,
                'f': self.f,
                'p': self.p,
                'r': self.r,
                'd_f': self.d_f,
                'd_p': self.d_p,
                'd_r': self.d_r,
                'f_': f * 100,
                'p_': prec * 100,
                'r_': recall * 100,
                'd_f_': f_d * 100,
                'd_p_': prec_d * 100,
                'd_r_': recall_d * 100
                }

    def compute(self, test=False, epoch_num=-1):
        # 同步outputs
        super().compute()
        self._write_result_to_file(test=test)
        mode = 'test' if test else 'val'
        gold_file = self.fields.conf.test_const if test else self.fields.conf.dev_const
        src_file = os.getcwd() + f"/{self.prefix}_output_{mode}.discbracket"
        p, r, f = call_eval(gold_file, src_file, self.fields.conf.proper_prm)
        d_p, d_r, d_f = call_eval(gold_file, src_file, self.fields.conf.proper_prm, disconly=True)
        self.p += p
        self.r += r
        self.f += f
        self.d_p += d_p
        self.d_r += d_r
        self.d_f += d_f

        return self.result



    def _write_result_to_file(self, test=False):
        mode = 'test' if test else 'val'
        outputs = self.outputs
        ids = [output['id'] for output in outputs]
        # raw_word = [output['raw_word'] for output in outputs]
        # arc_preds = [output['arc_preds'] for output in outputs]
        pred_trees = [output['pred_tree'] for output in outputs]

        total_len = sum(batch.shape[0] for batch in ids)

        final_results = [None for _ in range(total_len)]

        for batch in zip(ids, pred_trees):
            batch_ids, batch_pred_trees = batch
            for i in range(batch_ids.shape[0]):
                # recall that the first token is the imaginary root;
                final_results[batch_ids[i]] = batch_pred_trees[i]

        with open(f"{self.prefix}_output_{mode}.discbracket", 'w', encoding='utf8') as f:
            for tree in final_results:
                f.write(tree)
                f.write('\n')


def call_eval(gold_file, pred_file, cfg_file, disconly=False):
    """Just calls discodop eval and returns Prec, Recall, Fscore as floats"""
    # pdb.set_trace()
    params = ["discodop", "eval", gold_file, pred_file, cfg_file, "--fmt=discbracket"]
    if disconly:
        params.append("--disconly")

    result = subprocess.check_output(params)
    result = str(result).split("\\n")
    recall = result[-6].split()[-1]
    prec = result[-5].split()[-1]
    fscore = result[-4].split()[-1]
    if "nan" not in [prec, recall, fscore]:
        return float(prec), float(recall), float(fscore)
    return 0, 0, 0

