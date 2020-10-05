import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from torch.nn.utils import rnn
from torch.autograd import Variable
from GPG.models.model_utils import get_weights, get_act
from torch.nn.parameter import Parameter
from GPG.models.model_utils import init_lstm_wt, init_linear_wt



def tok_to_ent(tok2ent):
    if tok2ent == 'mean':
        return MeanPooling
    elif tok2ent == 'mean_max':
        return MeanMaxPooling
    else:
        raise NotImplementedError


def mean_pooling(input, mask):
    mean_pooled = input.sum(dim=1) / mask.sum(dim=1, keepdim=True)
    return mean_pooled


class MeanPooling(nn.Module):
    def __init__(self):
        super(MeanPooling, self).__init__()

    def forward(self, doc_state, entity_mapping, entity_lens):
        entity_states = entity_mapping.unsqueeze(3) * doc_state.unsqueeze(1)  # N x E x L x d
        mean_pooled = torch.sum(entity_states, dim=2) / entity_lens.unsqueeze(2)
        return mean_pooled


class MeanMaxPooling(nn.Module):
    def __init__(self):
        super(MeanMaxPooling, self).__init__()

    def forward(self, doc_state, entity_mapping, entity_lens):
        """
        :param doc_state:  N x L x d
        :param entity_mapping:  N x E x L
        :param entity_lens:  N x E
        :return: N x E x 2d
        """
        entity_states = entity_mapping.unsqueeze(3) * doc_state.unsqueeze(1)  # N x E x L x d
        max_pooled = torch.max(entity_states, dim=2)[0]
        mean_pooled = torch.sum(entity_states, dim=2) / entity_lens.unsqueeze(2)
        output = torch.cat([max_pooled, mean_pooled], dim=2)  # N x E x 2d
        return output


class GATSelfAttention(nn.Module):
    def __init__(self, in_dim, out_dim, config, layer_id=0, head_id=0):
        """ One head GAT """
        super(GATSelfAttention, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dropout = config.gnn_drop
        self.q_attn = config.q_attn
        self.query_dim = in_dim
        self.n_type = config.n_type

        # Case study
        self.layer_id = layer_id
        self.head_id = head_id
        self.step = 0

        self.W_type = nn.ParameterList()
        self.a_type = nn.ParameterList()
        self.qattn_W1 = nn.ParameterList()
        self.qattn_W2 = nn.ParameterList()
        for i in range(self.n_type):
            self.W_type.append(get_weights((in_dim, out_dim)))
            self.a_type.append(get_weights((out_dim * 2, 1)))

            if config.q_attn:
                q_dim = config.hidden_dim * 2
                self.qattn_W1.append(get_weights((q_dim, out_dim * 2)))
                self.qattn_W2.append(get_weights((out_dim * 2, out_dim * 2)))

        self.act = get_act('lrelu:0.2')

    def forward(self, input_state, adj, entity_mask, adj_mask=None, query_vec=None):
        zero_vec = torch.zeros_like(adj)
        scores = 0

        for i in range(self.n_type):
            h = torch.matmul(input_state, self.W_type[i])
            h = F.dropout(h, self.dropout, self.training)
            N, E, d = h.shape

            a_input = torch.cat([h.repeat(1, 1, E).view(N, E * E, -1), h.repeat(1, E, 1)], dim=-1)
            a_input = a_input.view(-1, E, E, 2*d)

            if self.q_attn:
                q_gate = F.relu(torch.matmul(query_vec, self.qattn_W1[i]))
                q_gate = torch.sigmoid(torch.matmul(q_gate, self.qattn_W2[i]))
                a_input = a_input * q_gate[:, None, None, :]
                score = self.act(torch.matmul(a_input, self.a_type[i]).squeeze(3))
            else:
                score = self.act(torch.matmul(a_input, self.a_type[i]).squeeze(3))
            scores += torch.where(adj == i+1, score, zero_vec)

        zero_vec = -9e15 * torch.ones_like(scores)
        scores = torch.where(adj > 0, scores, zero_vec)

        # Ahead Alloc
        if adj_mask is not None:
            h = h * adj_mask

        coefs = F.softmax(scores, dim=2)  # N * E * E
        h = coefs.unsqueeze(3) * h.unsqueeze(2)  # N * E * E * d
        h = torch.sum(h, dim=1)
        return h


class AttentionLayer(nn.Module):
    def __init__(self, in_dim, hid_dim, n_head, config, layer_id=0):
        super(AttentionLayer, self).__init__()
        assert hid_dim % n_head == 0
        self.dropout = config.gnn_drop

        self.attn_funcs = nn.ModuleList()
        for i in range(n_head):
            self.attn_funcs.append(  # in_dim = 
                GATSelfAttention(in_dim=in_dim, out_dim=hid_dim // n_head, config=config, layer_id=layer_id, head_id=i))

        if in_dim != hid_dim:
            self.align_dim = nn.Linear(in_dim, hid_dim)
            nn.init.xavier_uniform_(self.align_dim.weight, gain=1.414)
        else:
            self.align_dim = lambda x: x

    def forward(self, input, adj, entity_mask, adj_mask=None, query_vec=None):
        hidden_list = []
        for attn in self.attn_funcs:
            h = attn(input, adj, entity_mask, adj_mask=adj_mask, query_vec=query_vec)
            hidden_list.append(h)

        h = torch.cat(hidden_list, dim=-1)
        h = F.dropout(h, self.dropout, training=self.training)
        h = F.relu(h)
        return h



class ReduceState(nn.Module):
    def __init__(self, config):
        super(ReduceState, self).__init__()
        self.config = config
        self.encoder_num_layers = self.config.encoder_num_layers # 2
        self.decoder_num_layers = self.config.decoder_num_layers  # = 2
        self.reduce_h = nn.Linear(config.hidden_dim * 2 * self.encoder_num_layers , self.config.hidden_dim * 1 * self.decoder_num_layers, bias = False)
        init_linear_wt(self.reduce_h)
        self.reduce_c = nn.Linear(config.hidden_dim * 2 * self.encoder_num_layers, self.config.hidden_dim * 1 * self.decoder_num_layers, bias = False)
        init_linear_wt(self.reduce_c)

    def forward(self, hidden):
        h, c = hidden # h, c dim = 2 x b x hidden_dim
        # print(h.size())
        h_in = h.transpose(0, 1).contiguous().view(-1, self.config.hidden_dim * 2 * self.encoder_num_layers)
        hidden_reduced_h = F.relu(self.reduce_h(h_in))
        hidden_reduced_h = hidden_reduced_h.unsqueeze(0).contiguous().view(-1, self.decoder_num_layers * 1, self.config.hidden_dim)
        c_in = c.transpose(0, 1).contiguous().view(-1, self.config.hidden_dim * 2 * self.encoder_num_layers)
        hidden_reduced_c = F.relu(self.reduce_c(c_in))
        hidden_reduced_c = hidden_reduced_c.unsqueeze(0).contiguous().view(-1, self.decoder_num_layers * 1, self.config.hidden_dim)
        return (hidden_reduced_h.transpose(0, 1).contiguous(), hidden_reduced_c.transpose(0, 1).contiguous()) # h, c dim = 2 x b x hidden_dim

class Reason_Ans_InterBlocks(nn.Module):
    def __init__(self, config):
        super(Reason_Ans_InterBlocks, self).__init__()
        self.num_reason_layers = config.num_reason_layers
        self.reason_layer = ReasonLayer(doc_dim = config.hidden_dim * 2, config = config)
        self.ans_updata_layer = AnsUpdateLayer(doc_dim = config.hidden_dim * 2, ans_dim = config.hidden_dim * 2, config = config)

    def forward(self, doc_encoder_outputs, ans_encoder_outputs, batch):
        softmasks = []
        for l in range(self.num_reason_layers - 1):
            gated_u, u_outputs = self.reason_layer(doc_encoder_outputs, ans_encoder_outputs)
            doc_encoder_outputs_update = u_outputs if l == 0 else gated_u
            ans_state_update, softmask = self.ans_updata_layer(doc_encoder_outputs_update, ans_encoder_outputs, batch)
            softmasks.append(softmask)
            doc_encoder_outputs = doc_encoder_outputs_update
            ans_encoder_outputs = ans_state_update
        # l = l + 1
        gated_u, u_outputs = self.reason_layer(doc_encoder_outputs, ans_encoder_outputs)
        doc_encoder_outputs_update = gated_u
        return doc_encoder_outputs_update, softmasks

class AnsUpdateLayer(nn.Module):
    def __init__(self, doc_dim, ans_dim, config):
        super(AnsUpdateLayer, self).__init__()
        if config.tok2ent == 'mean_max':   # mean pooling and max pooling
            input_dim = doc_dim * 2
        else:
            input_dim = doc_dim
        self.tok2ent = tok_to_ent(config.tok2ent)()
        self.ans_weight = get_weights((ans_dim, input_dim))
        self.temp = np.sqrt(ans_dim * input_dim)
        self.gat = AttentionLayer(input_dim, doc_dim, config.n_heads, config)
        self.answer_update_layer = BiAttention(ans_dim, doc_dim, ans_dim, config.bi_attn_drop)
        self.answer_linear_layer = nn.Linear(ans_dim * 4, ans_dim)

    def forward(self, doc_state, ans_state, batch):
        entity_mapping = batch['entity_mapping']
        entity_length = batch['entity_lens']
        entity_mask = batch['entity_mask']
        adj = batch['entity_graphs']
        answer_mapping = batch['ans_mask']
        trunc_ans_mapping = answer_mapping[:, :50]
        trunc_ans_state = ans_state[:, :50]

        # if config.use_cuda:
        if True:   # cuda available
            entity_mapping = entity_mapping.cuda()
            entity_length = entity_length.cuda()
            entity_mask = entity_mask.cuda()
            adj = adj.cuda()
            trunc_ans_mapping = trunc_ans_mapping.cuda()
            trunc_ans_state = trunc_ans_state.cuda()

        entity_state = self.tok2ent(doc_state, entity_mapping, entity_length)
        ans_vec = mean_pooling(trunc_ans_state, trunc_ans_mapping)

        answer = torch.matmul(ans_vec, self.ans_weight)
        answer_scores = torch.bmm(entity_state, answer.unsqueeze(2)) / self.temp
        softmask = answer_scores * entity_mask.unsqueeze(2)  # N x E x 1  BCELossWithLogits
        adj_mask = torch.sigmoid(softmask)

        entity_state = self.gat(entity_state, adj, entity_mask, adj_mask=adj_mask, query_vec=ans_vec)
        ans_attn_output, _ = self.answer_update_layer(trunc_ans_state, entity_state, entity_mask)
        ans_state_update = self.answer_linear_layer(ans_attn_output)

        return ans_state_update, softmask

class ReasonLayer(nn.Module):
    def __init__(self, doc_dim, config):
        super(ReasonLayer, self).__init__()
        self._w_u = Parameter(torch.randn(doc_dim, 1, requires_grad=True))
        self._w_c = Parameter(torch.randn(2 * doc_dim, 1, requires_grad=True))
        self._w_r = Parameter(torch.randn(doc_dim, 1, requires_grad=True))
        self._b = Parameter(torch.randn(1, requires_grad=True))

        self.hidden_size = doc_dim

        self.doc_bilstm_layer = nn.LSTM(doc_dim * 3, doc_dim, num_layers=config.num_layers, dropout = config.dropout, batch_first=True)   # hidden_dim = 256
        init_lstm_wt(self.doc_bilstm_layer, 0.02)

    def forward(self, doc_encoder_outputs, ans_encoder_outputs):
        gate = None
        batch_size = doc_encoder_outputs.size(0)
        self.doc_bilstm_layer.flatten_parameters()
        # B x n x d
        _R = doc_encoder_outputs
        # B x m x d
        _C = ans_encoder_outputs
        _S = torch.bmm(_R, _C.transpose(1, 2))
        _H = torch.bmm(_R.transpose(1, 2), torch.softmax(_S, dim=-1))
        _G0 = torch.cat((_C.transpose(1, 2), _H), dim=1)  # B x 2*d x m
        _G = torch.bmm(_G0, torch.softmax(_S.transpose(1, 2), dim=-1))  # B x 2*d x n
        u_inp = torch.cat((_R, _G.transpose(1, 2)), dim=-1)  # B x n x 3*d
        u_outputs, u_hidden = self.doc_bilstm_layer(u_inp)   # B x n x d

        item_a = torch.mm(u_outputs.contiguous().view(-1, self.hidden_size), self._w_u)\
                        .contiguous()\
                        .view(batch_size, -1)   # B x n
        item_b = torch.mm(_G.transpose(1, 2).contiguous().view(-1, 2 * self.hidden_size), self._w_c)\
                        .contiguous()\
                        .view(batch_size, -1)   # B x n
        item_c = torch.mm(_R.contiguous().view(-1, self.hidden_size), self._w_r)\
                        .contiguous()\
                        .view(batch_size, -1) # B x n
        gates = torch.sigmoid((item_a + item_b + item_c + self._b))
        gates = gates.unsqueeze(-1) # B x n x 1
        gated_u = gates * _R + (1 - gates) * u_outputs   # B x n x d

        return gated_u, u_outputs


class BiAttention(nn.Module):
    def __init__(self, input_dim, memory_dim, hid_dim, dropout):
        super(BiAttention, self).__init__()
        self.dropout = dropout
        self.input_linear_1 = nn.Linear(input_dim, 1, bias=False)
        self.memory_linear_1 = nn.Linear(memory_dim, 1, bias=False)

        self.input_linear_2 = nn.Linear(input_dim, hid_dim, bias=True)
        self.memory_linear_2 = nn.Linear(memory_dim, hid_dim, bias=True)

        self.dot_scale = np.sqrt(input_dim)

    def forward(self, input, memory, mask):
        """
        :param input: context_encoding N * Ld * d
        :param memory: query_encoding N * Lm * d
        :param mask: query_mask N * Lm
        :return:
        """
        bsz, input_len, memory_len = input.size(0), input.size(1), memory.size(1)

        input = F.dropout(input, self.dropout, training=self.training)  # N x Ld x d
        memory = F.dropout(memory, self.dropout, training=self.training)  # N x Lm x d

        input_dot = self.input_linear_1(input)  # N x Ld x 1
        memory_dot = self.memory_linear_1(memory).view(bsz, 1, memory_len)  # N x 1 x Lm
        # N * Ld * Lm
        cross_dot = torch.bmm(input, memory.permute(0, 2, 1).contiguous()) / self.dot_scale
        # [f1, f2]^T [w1, w2] + <f1 * w3, f2>
        # (N * Ld * 1) + (N * 1 * Lm) + (N * Ld * Lm)
        att = input_dot + memory_dot + cross_dot  # N x Ld x Lm
        # N * Ld * Lm
        att = att - 1e30 * (1 - mask[:, None])

        input = self.input_linear_2(input)   # N x Ld x hid_dim
        memory = self.memory_linear_2(memory)   # N x Lm x hid_dim

        weight_one = F.softmax(att, dim=-1)   # N x Ld x Lm
        output_one = torch.bmm(weight_one, memory)   # N x Ld x hid_dim
        weight_two = F.softmax(att.max(dim=-1)[0], dim=-1).view(bsz, 1, input_len)   # N x 1 x Ld 
        output_two = torch.bmm(weight_two, input)  # N x 1 x hid_dim
        # N x Ld x hid_dim + N x Ld x hid_dim + N x Ld x hid_dim + N x Ld x hid_dim
        return torch.cat([input, output_one, input*output_one, output_two*output_one], dim=-1), memory

