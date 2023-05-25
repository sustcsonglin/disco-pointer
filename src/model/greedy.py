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

def identity(x):
    return x

class SpanSelection(nn.Module):
    def __init__(self, conf, fields):
        super(SpanSelection, self).__init__()
        self.conf = conf
        self.fields = fields
        self.vocab = self.fields.get_vocab('chart')

        self.metric = hydra.utils.instantiate(conf.metric.target, conf.metric, fields=fields, _recursive_=False)

        self.embeder = Embeder(fields=self.fields, conf=self.conf.embeder)

        # self.decoder_linear = MLP(4 * self.conf.n_lstm_hidden + self.conf.label_emb_size, self.embeder.get_output_dim(), 0.33)
        self.decoder_linear = identity

        input_size = 0
        output_size = 2 * self.conf.n_lstm_hidden if self.conf.encoder_type == 'LSTM' else 2 * self.embeder.get_output_dim()

        input_size += output_size
        input_size += self.conf.label_emb_size

        self.decoder_input_size = input_size
        self.output_size = self.conf.n_lstm_hidden

        from src.model.module.encoder.lstm_encoder import LSTMencoder
        self.encoder = LSTMencoder(self.conf.lstm_encoder, input_dim=self.embeder.get_output_dim())

        self.label_embedding = nn.Parameter(torch.rand(fields.get_vocab_size('chart'), conf.label_emb_size))

        self.mlp_prev_span = MLP(2 * output_size, output_size, dropout=conf.lstm_dropout)

        additional_size = 0

        self.biaffine = BiaffineScorer(n_in=output_size, n_out=self.conf.biaffine_size, bias_x=True, bias_y=True)
        # (n_in=self.conf.biaffine_size, bias_x=True, bias_y=True)

        self.label_projector = nn.Sequential(
            nn.Linear(2 * output_size , self.conf.label_emb_size),
            nn.LayerNorm(self.conf.label_emb_size),
            nn.ReLU(),
            nn.Linear(self.conf.label_emb_size, self.conf.label_emb_size),
        )


        self.criterion = nn.CrossEntropyLoss()
        self.span_criterion = nn.BCEWithLogitsLoss()
        self.start_emb = nn.Parameter(torch.randn(1, 2 * output_size))
        self.start_label = nn.Parameter(torch.randn(1, self.conf.label_emb_size))

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
        batch_size = repr.shape[0]

        s_span_score = self.biaffine(repr)

        seq_len = ctx['seq_len']
        max_seq_len = seq_len.max() + 1
        mask = seq_len.new_tensor(range(max_seq_len)) <= seq_len.view(-1, 1, 1)
        span_mask = mask & mask.new_ones(max_seq_len, max_seq_len).triu_(1)

        span_gold = span_mask.new_zeros(*span_mask.shape).float()
        span_gold[ ctx['gold_span'][:, 0], ctx['gold_span'][:, 1], ctx['gold_span'][:, 2]] = 1
        logits, gold = s_span_score[span_mask], span_gold[span_mask]
        span_loss  = self.span_criterion(logits, gold)

        ### remove [end of span]
        span_start = ctx['span_start'][:,:-1]
        span_end = ctx['span_end'][:,:-1]
        label = ctx['chart'][:, :-1]
        action_len = ctx['action_len'] - 1

        word_mask = torch.arange(seq_len.max(), device=repr.device)[None, None, :] >= span_start[:, :, None]
        word_mask = word_mask & (torch.arange(seq_len.max(), device=repr.device)[None, None, :] < span_end[:, :, None])
        max_pool_repr = \
        word_repr.unsqueeze(1).expand(batch_size, word_mask.shape[1], seq_len.max(), -1).clone().masked_fill_(
            ~word_mask.unsqueeze(-1), -1e3).max(-2)[0]

        end_repr = repr.gather(1, span_end.unsqueeze(-1).expand(*span_end.shape, repr.shape[-1]))
        start_repr = repr.gather(1, span_start.unsqueeze(-1).expand(*span_start.shape, repr.shape[-1]))

        gold_repr = end_repr - start_repr
        gold_repr = torch.cat([gold_repr, max_pool_repr], dim=-1)
        # batch_size * action_length * sen_len
        # normalization

        label_context = gold_repr
        label_logits = torch.matmul(self.label_projector(label_context), self.label_embedding.transpose(-1, -2))
        action_mask = torch.arange(action_len.max(), device=repr.device)[None, :] < action_len[:, None]
        label_logits, label_gold = label_logits[action_mask], label[action_mask]
        label_loss = self.criterion(label_logits, label_gold)

        # stop_logit = self.stop_mlp(output)
        # stop_label = action_mask.new_zeros(*action_mask.shape).long()
        # stop_label[torch.arange(batch_size), action_len-1] = 1
        # stop_logit, stop_label = stop_logit[action_mask], stop_label[action_mask]
        # stop_loss = self.criterion(stop_logit, stop_label)

        return (span_loss + label_loss)

    def decode(self, x, y):
        ctx = {**x, **y}
        repr, hx, word_repr = self.forward(ctx)
        # repr_l = self.mlp_src_l(repr)
        # repr_r = self.mlp_src_r(repr)
        batch_size = repr.shape[0]
        # mask out invalid positions


        # (batch_size, num_hyp, max_steps) maintain the partial parse results.

        seq_len = ctx['seq_len']
        max_seq_len = seq_len.max() + 1
        mask = seq_len.new_tensor(range(max_seq_len)) <= seq_len.view(-1, 1, 1)
        span_mask = mask & mask.new_ones(max_seq_len, max_seq_len).triu_(1)

        s_span_score = self.biaffine(repr)
        span_prediction = ((s_span_score.sigmoid() > 0.5) & span_mask)
        num_spans = span_prediction.sum([-1, -2])

        max_steps = int((2*seq_len.max() - 1)*2)
        predicted_span_left = repr.new_zeros(batch_size, max_steps).long()
        predicted_span_right = repr.new_zeros(batch_size, max_steps).long()
        predicted_span_label = repr.new_zeros(batch_size, max_steps).long()


        for b_idx in range(batch_size):
            prediction = span_prediction[b_idx].nonzero().tolist()
            for p_idx, (left, right) in enumerate(prediction):
                try:
                    assert left < right
                    word_mask = span_mask.new_zeros(word_repr.shape[1])
                    word_mask[left:right] = True
                    max_pool = word_repr[b_idx][word_mask].max(-2)[0]
                    gold_repr = torch.cat([repr[b_idx][right] - repr[b_idx][left],
                                           max_pool
                                           ], dim=-1
                                          )
                    label = (self.label_projector(gold_repr) @ self.label_embedding.transpose(-1, -2)).argmax(-1)
                except:
                    print("?")
                    pdb.set_trace()


                predicted_span_left[b_idx, p_idx] = left
                predicted_span_right[b_idx, p_idx] = right
                predicted_span_label[b_idx, p_idx] = label


        predicted_span_left = predicted_span_left.cpu().numpy()
        predicted_span_right = predicted_span_right.cpu().numpy()
        predicted_span_label = predicted_span_label.cpu().numpy()
        raw_word = ctx['raw_word']
        raw_pos = ctx['raw_pos']


        # recover constituent spans.
        results = []
        for b_idx in range(batch_size):
            span_num = int(num_spans[b_idx])
            left = predicted_span_left[b_idx, :span_num]
            right = predicted_span_right[b_idx, :span_num]
            label = predicted_span_label[b_idx, :span_num]
            postorder_sort = np.lexsort((-left, right))
            left = left[postorder_sort]
            right = right[postorder_sort]
            label = label[postorder_sort]
            # a simple transition system to resolve overlap.
            decoded_span = []
            cursor = []
            sent_len = int(seq_len[b_idx])
            lookup = [None for _ in range(sent_len + 1)]

            subtree = [Token(raw_word[b_idx][i], i, raw_pos[b_idx][i]) for i in range(sent_len)]

            for i in range(span_num):
                l = left[i]
                r = right[i]
                add_span = True

                for j in range(l + 1, r):
                    # overlap detection.
                    if lookup[j] is not None and lookup[j] < l:
                        add_span = False
                        break

                if lookup[r] is not None and lookup[r] == l:
                    add_span = False

                if add_span:
                    lab = self.vocab[label[i]]
                    children = []


                    for i in range(l, r):
                        if subtree[i] is not None:
                            children.append(subtree[i])
                            subtree[i] = None

                    new_node = Tree(lab, children)
                    subtree[r - 1] = new_node
                    if lookup[r] is None:
                        lookup[r] = l
                    else:
                        lookup[r] = min(l, lookup[r])

            child = []
            for i in range(sent_len):
                if subtree[i] is not None:
                    child.append(subtree[i])

            final_node = Tree('TOP', child)

            #     decoded_span.append([0, sent_len, 0])
            final_node.expand_unaries()
            results.append(final_node)
            assert len(final_node.span_sorted) == sent_len
        # pdb.set_trace()

        ctx['chart_preds'] = results

        return ctx



