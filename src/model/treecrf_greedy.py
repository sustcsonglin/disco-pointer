import pdb

import torch.nn as nn
import logging
import hydra

log = logging.getLogger(__name__)
from src.model.module.ember.embedding import Embeder
import torch
from supar.modules import MLP, Biaffine, Triaffine
from src.model.module.scorer.module.biaffine import BiaffineScorer
import numpy as np
from src.model.trees import Token, Tree
from supar.modules.dropout import SharedDropout
from .fn import stripe, diagonal_copy_, diagonal
import nltk
from functools import cmp_to_key


def compare(a, b):
    if a[1] > b[1]:
        return 1
    elif a[1] == b[1]:
        if a[0] > b[0]:
            return -1
        else:
            return 1
    else:

        return -1


@torch.enable_grad()
def cyk_decoding(s_span, lens):
    s_span = s_span.detach().clone().requires_grad_(True)
    batch, seq_len = s_span.shape[:2]
    s = s_span.new_zeros(batch, seq_len, seq_len).fill_(-1e9)
    s[:, torch.arange(seq_len-1), torch.arange(seq_len-1) + 1] = s_span[:, torch.arange(seq_len -1), torch.arange(seq_len-1) +1]

    for w in range(2, seq_len):
        n = seq_len - w
        left = stripe(s, n, w - 1, (0, 1))
        right = stripe(s, n, w - 1, (1, w), 0)
        composed = (left + right).max(2)[0]
        composed = composed + diagonal(s_span, w)
        diagonal_copy_(s, composed, w)
    logZ = s[torch.arange(batch), 0, lens]
    logZ.sum().backward()

    def get_post_order_span(predicted_span):
        predicted_span = predicted_span.detach().cpu().numpy()
        spans = []
        for i in range(predicted_span.shape[0]):
            span = predicted_span[i].nonzero()
            span.sort(key=cmp_to_key(compare))
            spans.append(span)

        return spans

    return get_post_order_span(s_span.grad)


@torch.enable_grad()
def cyk_partition(s_span, lens):
    # s_span = s_span.detach().clone().requires_grad_(True)

    batch, seq_len = s_span.shape[:2]
    s = s_span.new_zeros(batch, seq_len, seq_len).fill_(-1e9)
    s[:, torch.arange(seq_len-1), torch.arange(seq_len-1) + 1] = s_span[:, torch.arange(seq_len -1), torch.arange(seq_len-1) +1]

    for w in range(2, seq_len):
        n = seq_len - w
        left = stripe(s, n, w - 1, (0, 1))
        right = stripe(s, n, w - 1, (1, w), 0)
        composed = (left + right).logsumexp(2)
        composed = composed + diagonal(s_span, w)
        diagonal_copy_(s, composed, w)
    logZ = s[torch.arange(batch), 0, lens]
    return logZ



def identity(x):
    return x

class TreeCRF(nn.Module):
    def __init__(self, conf, fields):
        super(TreeCRF, self).__init__()
        self.conf = conf
        self.fields = fields
        self.vocab = self.fields.get_vocab('chart')

        self.metric = hydra.utils.instantiate(conf.metric.target, conf.metric, fields=fields, _recursive_=False)

        self.embeder = Embeder(fields=self.fields, conf=self.conf.embeder)

        # self.decoder_linear = MLP(4 * self.conf.n_lstm_hidden + self.conf.label_emb_size, self.embeder.get_output_dim(), 0.33)
        self.decoder_linear = identity

        output_size = 2 * self.conf.n_lstm_hidden

        from src.model.module.encoder.lstm_encoder import LSTMencoder
        self.encoder = LSTMencoder(self.conf.lstm_encoder, input_dim=self.embeder.get_output_dim())

        # self.label_embedding = nn.Parameter(torch.rand(fields.get_vocab_size('chart'), conf.label_emb_size))
        self.biaffine = BiaffineScorer(n_in=output_size, n_out=self.conf.biaffine_size, bias_x=True, bias_y=True)
        self.biaffine_label = BiaffineScorer(n_in=output_size, n_out=self.conf.biaffine_label_size,
                                             n_out_label=fields.get_vocab_size('chart'),
                                             bias_x=True, bias_y=True)
        self.criterion = nn.CrossEntropyLoss()


    def forward(self, ctx):
        # embeder
        self.embeder(ctx)
        self.encoder(ctx)
        # fencepost representations for boundaries.
        output = ctx['encoded_emb']
        x_f, x_b = output.chunk(2, -1)
        repr = torch.cat((x_f[:, :-1], x_b[:, 1:]), -1)
        hx = None
        return repr, hx, output[:, 1:-1]

    def get_loss(self, x, y):
        ctx = {**x, **y}

        repr, hx, word_repr = self.forward(ctx)

        s_span_score = self.biaffine(repr)
        s_label_score = self.biaffine_label(repr)
        seq_len = ctx['seq_len']
        gold = ctx['chart']
        logZ  = cyk_partition(s_span=s_span_score, lens=seq_len).sum()
        gold_score = s_span_score[gold[:, 0], gold[:, 1], gold[:, 2]].sum()

        span_loss = (logZ - gold_score)/seq_len.sum()
        label_loss = self.criterion(s_label_score[gold[:, 0], gold[:, 1], gold[:, 2]],  torch.tensor(gold[:, -1], device=logZ.device, dtype=torch.long))
        return (span_loss + label_loss)

    def decode(self, x, y):
        ctx = {**x, **y}
        repr, hx, word_repr = self.forward(ctx)
        # repr_l = self.mlp_src_l(repr)
        # repr_r = self.mlp_src_r(repr)
        batch_size = repr.shape[0]
        # mask out invalid positions

        seq_len = ctx['seq_len']
        s_span_score = self.biaffine(repr)
        s_label_score = self.biaffine_label(repr)

        result = cyk_decoding(s_span=s_span_score, lens=seq_len)


        def get_pred_charts(result, s_label):
            starts = result['start_idx']
            ends = result['end_idx']
            chart_preds = []
            for i in range(s_label.shape[0]):
                labels = s_label[i, starts[i], ends[i]].argmax(-1).tolist()
                chart_preds.append(list(zip(starts[i].tolist(), ends[i].tolist(), labels)))
            return chart_preds

        chart_preds = get_pred_charts(result, s_label_score)
        chart_preds = [[[span[0], span[1], span[2]] for span in chart] for chart in chart_preds]


        def build(tree, span):

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
                    if label == 'NULL' or '<>' in label:
                        return children
                    for sublabel in reversed(label.split("@")):
                        children = [nltk.Tree(sublabel, children)]

                return children


            children = helper()
            new = nltk.Tree("TOP", children)
            # try:
            assert len(new.pos()) == len(tree.pos())
            return new

        preds = [ [[i, j, self.vocab[k]] for i,j,k in tree]for tree in chart_preds]

        golds = [nltk.Tree.fromstring(tree) for tree in ctx['raw_tree']]
        preds = [
            build(tree, chart)
            for tree, chart in zip(golds, preds)
        ]
        ctx['chart_preds'] = preds
        return ctx


