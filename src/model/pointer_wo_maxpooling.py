import pdb

import torch.nn as nn
import logging
import hydra
log = logging.getLogger(__name__)
from src.model.module.ember.embedding import Embeder
import torch
from supar.modules import MLP, Biaffine, Triaffine
import numpy as np
from src.model.trees import Token, Tree
from supar.modules.dropout import SharedDropout

def identity(x):
    return x

class UniLSTMDec(nn.Module):
    def __init__(self,
                 input_dim,
                 output_dim,
                 dropout_rate):
        super(UniLSTMDec, self).__init__()
        self._rnn = nn.LSTM(input_dim, output_dim, batch_first=True)
        self._dropout = nn.Dropout(dropout_rate)

    def forward(self, hidden, hx=None, training=False):
        drop_h = self._dropout(hidden)
        if len(hidden.size()) == 2:
            drop_h = drop_h.unsqueeze(1)
        if hx is None:
            rnn_h, (nxt_s, nxt_c) = self._rnn(drop_h)
        else:
            state, cell = hx
            assert len(state.size()) == 2
            rnn_h, (nxt_s, nxt_c) = self._rnn(drop_h, (state.unsqueeze(0), cell.unsqueeze(0)))

        if not training:
            rnn_h = rnn_h.squeeze(1)

        return rnn_h, (nxt_s.squeeze(0), nxt_c.squeeze(0))


class PointerNet(nn.Module):
    def __init__(self, conf, fields):
        super(PointerNet, self).__init__()
        self.conf = conf
        self.fields = fields
        self.vocab = self.fields.get_vocab('chart')

        self.metric = hydra.utils.instantiate(conf.metric.target, conf.metric, fields=fields, _recursive_=False)

        self.embeder =  Embeder(fields=self.fields, conf=self.conf.embeder)

        # self.decoder_linear = MLP(4 * self.conf.n_lstm_hidden + self.conf.label_emb_size, self.embeder.get_output_dim(), 0.33)
        self.decoder_linear = identity

        input_size = 0
        output_size = 2 * self.conf.n_lstm_hidden if self.conf.encoder_type == 'LSTM' else 2 * self.embeder.get_output_dim()

        input_size +=  output_size
        input_size +=  self.conf.label_emb_size

        self.decoder_input_size = input_size
        self.output_size = self.conf.n_lstm_hidden

        from src.model.module.encoder.lstm_encoder import LSTMencoder
        self.encoder = LSTMencoder(self.conf.lstm_encoder, input_dim=self.embeder.get_output_dim())

        self.decoder = UniLSTMDec(input_dim= self.decoder_input_size, output_dim=self.conf.n_lstm_hidden, dropout_rate=self.conf.lstm_dropout)

        self.label_embedding = nn.Parameter(torch.rand(fields.get_vocab_size('chart'), conf.label_emb_size))


        self.mlp_prev_span = MLP(output_size, output_size, dropout=conf.lstm_dropout)

        additional_size = 0
        self.mlp_src_l = MLP(n_in=output_size, n_out=self.conf.biaffine_size, dropout=conf.lstm_dropout)
        self.mlp_src_r = MLP(n_in=output_size, n_out=self.conf.biaffine_size, dropout=conf.lstm_dropout)
        self.mlp_dec = MLP(n_in=self.conf.n_lstm_hidden + additional_size, n_out=self.conf.biaffine_size, dropout=conf.lstm_dropout)
        self.triaffine =  Triaffine(n_in=self.conf.biaffine_size)

        self.label_projector = nn.Sequential(
        nn.Linear(output_size+self.conf.n_lstm_hidden, self.conf.label_emb_size),
        nn.LayerNorm(self.conf.label_emb_size),
        nn.ReLU(),
        nn.Linear(self.conf.label_emb_size, self.conf.label_emb_size),
        )


        # self.stop_mlp = nn.Sequential(
        # nn.Linear(self.conf.n_lstm_hidden, self.conf.label_emb_size),
        # nn.LayerNorm(self.conf.label_emb_size),
        # nn.ReLU(),
        # nn.Linear(self.conf.label_emb_size, 2),
        # )



        self.criterion = nn.CrossEntropyLoss()
        self.start_emb = nn.Parameter(torch.randn(1, output_size))
        self.start_label = nn.Parameter(torch.randn(1, self.conf.label_emb_size))



    def forward(self, ctx):
        # embeder

        self.embeder(ctx)

        self.encoder(ctx)
            # fencepost representations for boundaries.
        output = ctx['encoded_emb']
        x_f, x_b = output.chunk(2, -1)
        repr =  torch.cat((x_f[:, :-1], x_b[:, 1:]), -1)
        hx = None
        return repr, hx, output[:, 1:-1]

    def get_loss(self, x, y):
        ctx = {**x, **y}

        repr,  hx, word_repr = self.forward(ctx)
        batch_size = repr.shape[0]
        biaffine_src_l_repr = self.mlp_src_l(repr)
        biaffine_src_r_repr = self.mlp_src_r(repr)

        seq_len = ctx['seq_len']
        max_seq_len = seq_len.max() + 1
        mask = seq_len.new_tensor(range(max_seq_len)) <= seq_len.view(-1, 1, 1)
        span_mask = mask & mask.new_ones(max_seq_len, max_seq_len).triu_(1)
        # end.
        span_mask[:, 0, 0] = True

        span_start = ctx['span_start']
        span_end = ctx['span_end']
        word_mask = torch.arange(seq_len.max(), device=repr.device)[None, None, :] >= span_start[:, :, None]
        word_mask = word_mask & (torch.arange(seq_len.max(), device=repr.device)[None, None, :] < span_end[:, :, None])

        # max_pool_repr = word_repr.unsqueeze(1).expand(batch_size, word_mask.shape[1] , seq_len.max(), -1).clone().masked_fill_(
            # ~word_mask.unsqueeze(-1), -1e3).max(-2)[0]

        label = ctx['chart']


        end_repr = repr.gather(1, span_end.unsqueeze(-1).expand(*span_end.shape, repr.shape[-1]))
        start_repr = repr.gather(1, span_start.unsqueeze(-1).expand(*span_start.shape, repr.shape[-1]))
        gold_repr = end_repr - start_repr

        # gold_repr = torch.cat([gold_repr, max_pool_repr], dim=-1)

        decoder_input_emb = []

        decoder_input_emb.append(
                self.mlp_prev_span(
                torch.cat([ self.start_emb.unsqueeze(0).expand(batch_size, 1, 2*repr.shape[-1]), gold_repr[:, :-1]], dim=1)
                )
        )

        prev_label =  label[:, :-1]
        decoder_input_emb.append(
                torch.cat(
                    [self.start_label.unsqueeze(0).expand(batch_size, 1, self.conf.label_emb_size), self.label_embedding[prev_label]]
                , 1)
        )

        action_len = ctx['action_len']
        decoder_input = torch.cat(decoder_input_emb, dim=-1)

        output, _ = self.decoder(decoder_input,  hx, training=True)

        output_biaffine = self.mlp_dec(output)
        logits = self.triaffine.forward_bmx_bny_baz_2_bamn(biaffine_src_l_repr, biaffine_src_r_repr, output_biaffine)
        #TODO:
        logits = logits.masked_fill_(~span_mask.unsqueeze(1), float("-inf")).flatten(start_dim=-2)
        # masked out invalid positions.
        # logits = logits.masked_fill_(torch.arange(seq_len.max()+1, device=repr.device)[None, None, :] > seq_len[:, None, None], float('-inf'))
        action_mask = torch.arange(action_len.max(), device=repr.device)[None, :] < action_len[:, None]
        # batch_size * action_length * sen_len
        # normalization
        logits, action_gold = logits[action_mask], (span_start*(seq_len.max()+1) + span_end)[action_mask]
        pointer_loss = self.criterion(logits, action_gold)

        label_context = torch.cat([gold_repr, output], dim=-1)
        label_logits =  torch.matmul(self.label_projector(label_context), self.label_embedding.transpose(-1, -2))
        label_logits, label_gold = label_logits[action_mask], label[action_mask]
        label_loss = self.criterion(label_logits, label_gold)

        # stop_logit = self.stop_mlp(output)
        # stop_label = action_mask.new_zeros(*action_mask.shape).long()
        # stop_label[torch.arange(batch_size), action_len-1] = 1
        # stop_logit, stop_label = stop_logit[action_mask], stop_label[action_mask]
        # stop_loss = self.criterion(stop_logit, stop_label)

        return (pointer_loss + label_loss )




    def decode(self, x, y):
        ctx = {**x, **y}
        repr, hx,word_repr = self.forward(ctx)
        repr_l = self.mlp_src_l(repr)
        repr_r = self.mlp_src_r(repr)
        batch_size = repr.shape[0]
        seq_len = ctx['seq_len']
        # mask out invalid positions


        # (batch_size, num_hyp, max_steps) maintain the partial parse results.

        seq_len = ctx['seq_len']
        max_seq_len = seq_len.max() + 1
        mask = seq_len.new_tensor(range(max_seq_len)) <= seq_len.view(-1, 1, 1)
        span_mask = mask & mask.new_ones(max_seq_len, max_seq_len).triu_(1)
        span_mask[:, 0, 0] = True


        len_boundary = seq_len.max()+ 1
        max_steps = int((2*seq_len.max() - 1)*2)

        predicted_span_left = repr.new_zeros(batch_size, max_steps).long()
        predicted_span_right = repr.new_zeros(batch_size, max_steps).long()
        predicted_span_label = repr.new_zeros(batch_size, max_steps).long()
        is_finish =  repr.new_zeros(batch_size).bool()
        num_spans = (2*seq_len - 1) * 2

        decoder_input_emb = torch.cat([
            self.mlp_prev_span(self.start_emb.expand(batch_size,  2*repr.shape[-1])),
            self.start_label.expand(batch_size,  self.conf.label_emb_size)
        ],-1)


        for step in range(max_steps):
            out, hx = self.decoder(decoder_input_emb, hx)
            out_biaffine = self.mlp_dec(out)
            point_span_score = self.triaffine.forward_bmx_bny_baz_2_bamn(repr_l, repr_r, out_biaffine.unsqueeze(-2)).squeeze(-3)
            next_span  = point_span_score.masked_fill_(~span_mask, float('-inf')).flatten(start_dim=-2).argmax(-1)
            next_span_start = (next_span / len_boundary).long()
            next_span_end = next_span % len_boundary

            # assert (next_span_start < next_span_end) | ()


            next_span_start_repr = repr.gather(1, next_span_start[:, None, None].expand(batch_size, 1, repr.shape[-1])).squeeze(1)
            next_span_end_repr = repr.gather(1, next_span_end[:, None, None].expand(batch_size, 1, repr.shape[-1])).squeeze(1)
            next_span_repr =  next_span_end_repr - next_span_start_repr

            word_mask = torch.arange(seq_len.max(), device=decoder_input_emb.device)[None, :] < next_span_end[:, None]
            word_mask = word_mask & (torch.arange(seq_len.max(), device=decoder_input_emb.device)[None, :] >= next_span_start[:, None])
            
            # max_pool_repr = word_repr.clone().masked_fill_(
                # ~word_mask.unsqueeze(-1), -1e3).max(-2)[0
            # next_span_repr = torch.cat([next_span_repr, max_pool_repr], dim=-1)


            next_span_label =  (self.label_projector(torch.cat([next_span_repr, out], dim=-1)) @ self.label_embedding.t()).argmax(-1)

            decoder_input_emb = torch.cat(
                [self.mlp_prev_span(next_span_repr),
                 self.label_embedding[next_span_label]
                ], dim=-1
            )

            # stop = self.stop_mlp(out).argmax(-1)



            new_is_finish = ((next_span_start == 0) & (next_span_end == 0))

            num_spans[new_is_finish & ~is_finish] = step

            is_finish = is_finish |  new_is_finish
            predicted_span_left[:, step] = next_span_start
            predicted_span_right[:, step] = next_span_end
            predicted_span_label[:, step] = next_span_label

            # early exist
            if is_finish.all():
                break


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
            postorder_sort = np.lexsort((-left,right))
            left = left[postorder_sort]
            right = right[postorder_sort]
            label = label[postorder_sort]
            # a simple transition system to resolve overlap.
            decoded_span = []
            cursor = []
            sent_len = int(seq_len[b_idx])
            lookup = [None for _ in range(sent_len + 1)]

            subtree = [ Token(raw_word[b_idx][i], i, raw_pos[b_idx][i]) for i in range(sent_len)]

            for i in range(span_num):
                l = left[i]
                r = right[i]
                add_span = True

                for j in range(l+1, r):
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
                    subtree[r-1] = new_node
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




