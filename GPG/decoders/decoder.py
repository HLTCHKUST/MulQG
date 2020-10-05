import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from GPG.data import *
import copy
from torch_scatter import scatter_max
from GPG.models.model_utils import init_linear_wt, init_lstm_wt
# from GPG.data.vocab import Vocab
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
INF = 1e12

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "UNKNOWN"
START_TOKEN = "<s>"
END_TOKEN = "EOS"

PAD_ID = 0
UNK_ID = 1
START_ID = 2
END_ID = 3

class Attention(nn.Module):
    def __init__(self, config):
        super(Attention, self).__init__()
        # attention
        self.config = config
        if self.config.is_coverage:
            self.W_c = nn.Linear(1, self.config.hidden_dim * 2, bias=False)
        self.decode_proj = nn.Linear(self.config.hidden_dim * 2 * self.config.decoder_num_layers, self.config.hidden_dim * 2)
        self.v = nn.Linear(self.config.hidden_dim * 2, 1, bias=False)

    def forward(self, s_t_hat, encoder_outputs, encoder_feature, enc_padding_mask, coverage):
        b, t_k, n = list(encoder_outputs.size())
        
        dec_fea = self.decode_proj(s_t_hat) # B x 2*hidden_dim
        dec_fea_expanded = dec_fea.unsqueeze(1).expand(b, t_k, n).contiguous() # B x t_k x 2*hidden_dim
        dec_fea_expanded = dec_fea_expanded.view(-1, n)  # B * t_k x 2*hidden_dim

        att_features = encoder_feature + dec_fea_expanded # B * t_k x 2*hidden_dim
        if self.config.is_coverage:
            coverage_input = coverage.view(-1, 1)  # B * t_k x 1
            coverage_feature = self.W_c(coverage_input)  # B * t_k x 2*hidden_dim
            att_features = att_features + coverage_feature

        e = F.tanh(att_features) # B * t_k x 2*hidden_dim
        scores = self.v(e)  # B * t_k x 1
        scores = scores.view(-1, t_k)  # B x t_k

        attn_dist_ = F.softmax(scores, dim=1)*enc_padding_mask # B x t_k
        normalization_factor = attn_dist_.sum(1, keepdim=True)
        attn_dist = attn_dist_ / normalization_factor

        attn_dist = attn_dist.unsqueeze(1)  # B x 1 x t_k
        c_t = torch.bmm(attn_dist, encoder_outputs)  # B x 1 x n
        c_t = c_t.view(-1, self.config.hidden_dim * 2)  # B x 2*hidden_dim

        attn_dist = attn_dist.view(-1, t_k)  # B x t_k

        if self.config.is_coverage:
            coverage = coverage.view(-1, t_k)
            coverage = coverage + attn_dist

        return c_t, attn_dist, coverage


class Pointer_Decoder(nn.Module):
    def __init__(self, config, embeddings = None):
        super(Pointer_Decoder, self).__init__()
        self.config = config
        self.embeddings = nn.Embedding(config.vocab_size, config.emb_dim)
        if embeddings is not None:
            self.word_embeddings = nn.Embedding(config.vocab_size, config.emb_dim). \
                from_pretrained(embeddings, freeze=True)

        emb_dim = config.emb_dim

        self.lstm = nn.LSTM(emb_dim, self.config.hidden_dim, num_layers=self.config.decoder_num_layers, dropout = self.config.dropout, batch_first=True, bidirectional=False)
        # self.lstm = nn.LSTM(emb_dim, self.config.hidden_dim, num_layers=self.config.decoder_num_layers, dropout = self.config.dropout, batch_first=True, bidirectional=False)
        init_lstm_wt(self.lstm)

        self.x_context = nn.Linear(self.config.hidden_dim * 2 + emb_dim, emb_dim)
        self.p_gen_linear = nn.Linear(emb_dim + self.config.hidden_dim * 2 + self.config.hidden_dim * 2 * self.config.decoder_num_layers, 1)

        self.encoder_trans = nn.Linear(self.config.hidden_dim * 2, self.config.hidden_dim * 2, bias = False)

        self.out1 = nn.Linear(self.config.hidden_dim * 3, self.config.hidden_dim)
        self.out2 = nn.Linear(self.config.hidden_dim, self.config.vocab_size)
        init_linear_wt(self.out2)

        self.attention_network = Attention(self.config)

        # self.softmax = nn.Softmax(-1)
        # self.dropout = nn.Dropout(self.config.dropout)


    def get_encoder_features(self, encoder_outputs):
        encoder_feature = encoder_outputs.contiguous().view(-1, self.config.hidden_dim * 2)  # B * t_k x 2*hidden_dim
        return self.encoder_trans(encoder_feature)

    def forward(self, inputs, init_state, encoder_outputs, encoder_padding_mask, extra_zeros, enc_batch_extend_vocab, coverage):

        self.lstm.flatten_parameters()
        encoder_feature = self.get_encoder_features(encoder_outputs)

        context_0 = torch.zeros(inputs.size(0), 2 * self.config.hidden_dim)

        self.embeddings = self.embeddings.cuda()
        embs=self.embeddings(inputs)
        context = context_0
        outputs, state, attns, p_gens, coverages = [], init_state, [], [], []

        for emb in embs.split(1, dim=1):

            x = self.x_context(torch.cat((context, emb.squeeze(1)), 1))    # x should be batch_size x emb_dim

            output, state = self.lstm(x.unsqueeze(1), state)    # batch_size * 1 * emb_dim, state: 2 x b x hidden_dim
            h_decoder, c_decoder = state   # 2 x b x hidden_dim
            state_hat = torch.cat((h_decoder.view(-1, self.config.hidden_dim * self.config.decoder_num_layers),
                             c_decoder.view(-1, self.config.hidden_dim * self.config.decoder_num_layers)), 1)  # B x 4 * hidden_dim
            context, attn_dist, coverage_next = self.attention_network(state_hat, encoder_outputs, encoder_feature,
                                                          encoder_padding_mask, coverage)

            coverage = coverage_next
            p_gen = None
            if self.config.use_copy:
                p_gen_input = torch.cat((context, state_hat, x), 1)  # B x (2*hidden_dim + 2*2*hidden_dim + emb_dim)
                p_gen = self.p_gen_linear(p_gen_input)
                p_gen = F.sigmoid(p_gen)
            p_gens.append(p_gen)
            
            output = torch.cat((output.view(-1, self.config.hidden_dim), context), 1) # B x hidden_dim * 3
            output = self.out1(output) # B x hidden_dim
            # output = torch.tanh(self.out1(output))
            output = self.out2(output) # B x vocab_size
            vocab_dist = F.softmax(output, dim=1)
            
            assert not torch.isnan(torch.sum(attn_dist)).item(), attn_dist

            if self.config.use_copy:
                vocab_dist_ = p_gen * vocab_dist
                attn_dist_ = (1 - p_gen) * attn_dist
                if extra_zeros is not None:
                    vocab_dist_ = torch.cat([vocab_dist_, extra_zeros], 1)
                final_dist = vocab_dist_.scatter_add(1, enc_batch_extend_vocab, attn_dist_)
            else:
                final_dist = vocab_dist

            outputs += [final_dist]
            attns += [attn_dist]
            coverages += [coverage]
        outputs = torch.stack(outputs, 1)
        attns = torch.stack(attns, 1)
        p_gens = torch.stack(p_gens, 1)
        coverages = torch.stack(coverages, 1)
        return outputs, state, attns, p_gens, coverages

    # def sample(self, input, init_state, encoder_outputs, encoder_feature, encoder_padding_mask, context_0, extra_zeros, enc_batch_extend_vocab, coverage_0):
    #     inputs, outputs, sample_ids, state = [], [], [], init_state
    #     attns = []
    #     p_gens = []
    #     inputs += input
    #     max_time_step = self.config.max_tgt_len
    #     context = context_0
    #     coverage = coverage_0
    #     for i in range(max_time_step):
    #         output, state, attn_weights, p_gen, coverage, context = self.sample_one(inputs[i], state, encoder_outputs, encoder_feature,extra_zeros, encoder_padding_mask, context, enc_batch_extend_vocab, coverage)
    #         p_gens.append(p_gen)

    #         if(self.config.top_k_top_p):
    #             predicted = torch.zeros(output.size(0), dtype=torch.long).to(input[0].device)
    #             for i in range(output.size(0)):
    #                 output_filtered = top_k_top_p_filtering(output[i,:], top_k=3, top_p=0.9)
    #                 probabilities = F.softmax(output_filtered, dim=-1)
    #                 next_token = torch.multinomial(probabilities, 1)
    #                 predicted[i] = next_token.item()
    #         else:
    #             predicted = output.max(dim=1)[1]
           
    #         input_predicted = predicted.cpu().clone()
    #         for j in range(input_predicted.size()[0]):
    #             if input_predicted[j] >= self.vocab_size:
    #                 input_predicted[j] = torch.LongTensor([UNK])
    #         inputs += [input_predicted.to(input[0].device)]
    #         sample_ids += [predicted]
    #         outputs += [output]
    #         attns += [attn_weights]

    #     sample_ids = torch.stack(sample_ids, 1)
    #     attns = torch.stack(attns, 1)
    #     p_gens = torch.stack(p_gens, 1)
    #     return sample_ids, (outputs, attns), p_gens

    def sample_one(self, input, init_state, encoder_outputs, encoder_padding_mask, context_0, extra_zeros, enc_batch_extend_vocab, coverage):
        self.lstm.flatten_parameters()
        encoder_feature = self.get_encoder_features(encoder_outputs)

        assert input.max() < self.config.vocab_size, input
        self.embeddings = self.embeddings.cuda()
        emb=self.embeddings(input)
        x = self.x_context(torch.cat((context_0, emb.squeeze(1)), 1)) 
        output, state = self.lstm(x.unsqueeze(1), init_state)
        h_decoder, c_decoder = state  
        state_hat = torch.cat((h_decoder.view(-1, self.config.hidden_dim * self.config.decoder_num_layers), c_decoder.view(-1, self.config.hidden_dim * self.config.decoder_num_layers)), 1)  # B x 4 * hidden_dim

        context, attn_dist, coverage_next = self.attention_network(state_hat, encoder_outputs, encoder_feature,
                                                          encoder_padding_mask, coverage)
        p_gen_input = torch.cat((context, state_hat, x), 1) 
        p_gen = self.p_gen_linear(p_gen_input)
        p_gen = F.sigmoid(p_gen)

        output = torch.cat((output.view(-1, self.config.hidden_dim), context), 1) # B x hidden_dim * 3
        output = self.out1(output) # B x hidden_dim
        output = self.out2(output) # B x vocab_size
        vocab_dist = F.softmax(output, dim=1)
        # output = torch.tanh(self.out1(output))
        # vocab_dist = self.out2(output) # B x vocab_size
        assert not torch.isnan(torch.sum(attn_dist)).item(), attn_dist

        vocab_dist_ = p_gen * vocab_dist
        attn_dist_ = (1 - p_gen) * attn_dist
        if extra_zeros is not None:
            vocab_dist_ = torch.cat([vocab_dist_, extra_zeros], 1)
        final_dist = vocab_dist_.scatter_add(1, enc_batch_extend_vocab, attn_dist_)

        return final_dist, state, attn_dist, p_gen, coverage_next, context


class Pointer_Decoder_Base(nn.Module):
    def __init__(self, config, embeddings = None):
        super(Pointer_Decoder_Base, self).__init__()
        self.config = config

        self.embeddings = nn.Embedding(config.vocab_size, config.emb_dim)
        if embeddings is not None:
            self.word_embeddings = nn.Embedding(config.vocab_size, config.emb_dim). \
                from_pretrained(embeddings, freeze=True)

        emb_dim = config.emb_dim
        self.lstm = nn.LSTM(emb_dim, self.config.hidden_dim, num_layers=self.config.decoder_num_layers, dropout = self.config.dropout, batch_first=True, bidirectional=False)
        init_lstm_wt(self.lstm)

        self.x_context = nn.Linear(self.config.hidden_dim * 2 + emb_dim, emb_dim)
        self.p_gen_linear = nn.Linear(emb_dim + self.config.hidden_dim * 2 + self.config.hidden_dim * 2 * self.config.decoder_num_layers, 1)

        self.encoder_trans = nn.Linear(self.config.hidden_dim * 2, self.config.hidden_dim * 2, bias = False)

        self.out1 = nn.Linear(self.config.hidden_dim * 3, self.config.hidden_dim)
        self.out2 = nn.Linear(self.config.hidden_dim, self.config.vocab_size)
        init_linear_wt(self.out2)

        self.attention_network = Attention(self.config)

        # self.softmax = nn.Softmax(-1)
        # self.dropout = nn.Dropout(self.config.dropout)

    def get_encoder_features(self, encoder_outputs):
        encoder_feature = encoder_outputs.contiguous().view(-1, self.config.hidden_dim * 2)  # B * t_k x 2*hidden_dim
        return self.encoder_trans(encoder_feature)

    def forward(self, inputs, init_state, encoder_outputs, encoder_padding_mask,context_0, extra_zeros, enc_batch_extend_vocab, coverage):

        self.lstm.flatten_parameters()
        encoder_feature = self.get_encoder_features(encoder_outputs)

        self.embeddings = self.embeddings.cuda()
        embs=self.embeddings(inputs)
        context = context_0
        outputs, state, attns, p_gens, coverages = [], init_state, [], [], []

        for emb in embs.split(1, dim=1):
            x = self.x_context(torch.cat((context, emb.squeeze(1)), 1))    # x should be batch_size x emb_dim
            output, state = self.lstm(x.unsqueeze(1), state)    # batch_size * 1 * emb_dim, state: 2 x b x hidden_dim
            h_decoder, c_decoder = state   # 2 x b x 2*hidden_dim
            state_hat = torch.cat((h_decoder.view(-1, self.config.hidden_dim * self.config.decoder_num_layers),
                             c_decoder.view(-1, self.config.hidden_dim * self.config.decoder_num_layers)), 1)  # B x 4 * hidden_dim
            context, attn_dist, coverage_next = self.attention_network(state_hat, encoder_outputs, encoder_feature,
                                                          encoder_padding_mask, coverage)

            coverage = coverage_next
            p_gen = None
            if self.config.use_copy:
                p_gen_input = torch.cat((context, state_hat, x), 1)  # B x (2*hidden_dim + 2*2*hidden_dim + emb_dim)
                p_gen = self.p_gen_linear(p_gen_input)
                p_gen = F.sigmoid(p_gen)
            p_gens.append(p_gen)
            
            output = torch.cat((output.view(-1, self.config.hidden_dim), context), 1) # B x hidden_dim * 3
            # output = self.out1(output) # B x hidden_dim
            output = torch.tanh(self.out1(output))
            vocab_dist = self.out2(output) # B x vocab_size
            # vocab_dist = F.softmax(output, dim=1)
            
            assert not torch.isnan(torch.sum(attn_dist)).item(), attn_dist

            if self.config.use_copy:
                vocab_dist_ = p_gen * vocab_dist
                attn_dist_ = (1 - p_gen) * attn_dist
                if extra_zeros is not None:
                    vocab_dist_ = torch.cat([vocab_dist_, extra_zeros], 1)
                final_dist = vocab_dist_.scatter_add(1, enc_batch_extend_vocab, attn_dist_)
            else:
                final_dist = vocab_dist

            outputs += [final_dist]
            attns += [attn_dist]
            coverages += [coverage]
        outputs = torch.stack(outputs, 1)
        attns = torch.stack(attns, 1)
        p_gens = torch.stack(p_gens, 1)
        coverages = torch.stack(coverages, 1)
        return outputs, state, attns, p_gens, coverages

    def sample_one(self, input, init_state, encoder_outputs, encoder_padding_mask, context_0, extra_zeros, enc_batch_extend_vocab, coverage):
        self.lstm.flatten_parameters()
        encoder_feature = self.get_encoder_features(encoder_outputs)

        assert input.max() < self.config.vocab_size, input
        self.embeddings = self.embeddings.cuda()
        emb=self.embeddings(input)
        x = self.x_context(torch.cat((context_0, emb.squeeze(1)), 1)) 
        output, state = self.lstm(x.unsqueeze(1), init_state)
        h_decoder, c_decoder = state  
        state_hat = torch.cat((h_decoder.view(-1, self.config.hidden_dim * self.config.decoder_num_layers), c_decoder.view(-1, self.config.hidden_dim * self.config.decoder_num_layers)), 1)  # B x 4 * hidden_dim

        context, attn_dist, coverage_next = self.attention_network(state_hat, encoder_outputs, encoder_feature,
                                                          encoder_padding_mask, coverage)
        p_gen_input = torch.cat((context, state_hat, x), 1) 
        p_gen = self.p_gen_linear(p_gen_input)
        p_gen = F.sigmoid(p_gen)

        output = torch.cat((output.view(-1, self.config.hidden_dim), context), 1) # B x hidden_dim * 3
        # output = self.out1(output) # B x hidden_dim
        # output = self.out2(output) # B x vocab_size
        # vocab_dist = F.softmax(output, dim=1)

        # output = self.out1(output) # B x hidden_dim
        output = torch.tanh(self.out1(output))
        vocab_dist = self.out2(output) # B x vocab_size
            # vocab_dist = F.softmax(output, dim=1)
        
        assert not torch.isnan(torch.sum(attn_dist)).item(), attn_dist

        vocab_dist_ = p_gen * vocab_dist
        attn_dist_ = (1 - p_gen) * attn_dist
        if extra_zeros is not None:
            vocab_dist_ = torch.cat([vocab_dist_, extra_zeros], 1)
        final_dist = vocab_dist_.scatter_add(1, enc_batch_extend_vocab, attn_dist_)

        return final_dist, state, attn_dist, p_gen, coverage_next, context


class Decoder(nn.Module):
    def __init__(self, config, embeddings):
        super(Decoder, self).__init__()
        self.vocab_size = config.vocab_size
        embedding_size = config.emb_dim
        hidden_size = config.hidden_dim * 2
        num_layers = config.decoder_num_layers
        dropout = config.dropout

        self.embedding = nn.Embedding(self.vocab_size, embedding_size)
        if embeddings is not None:
            self.embedding = nn.Embedding(self.vocab_size, embedding_size). \
                from_pretrained(embeddings, freeze=True)

        if num_layers == 1:
            dropout = 0.0
        self.encoder_trans = nn.Linear(hidden_size, hidden_size)
        self.reduce_layer = nn.Linear(embedding_size + hidden_size, embedding_size)
        self.lstm = nn.LSTM(embedding_size, hidden_size, batch_first=True,
                            num_layers=num_layers, bidirectional=False, dropout=dropout)
        self.concat_layer = nn.Linear(2 * hidden_size, hidden_size)
        self.logit_layer = nn.Linear(hidden_size, self.vocab_size)

    @staticmethod
    def attention(query, memories, mask):
        # query : [b, 1, d]
        energy = torch.matmul(query, memories.transpose(1, 2))  # [b, 1, t]
        energy = energy.squeeze(1).masked_fill(mask, value=-1e12)
        attn_dist = F.softmax(energy, dim=1).unsqueeze(dim=1)  # [b, 1, t]
        context_vector = torch.matmul(attn_dist, memories)  # [b, 1, d]

        return context_vector, energy

    def get_encoder_features(self, encoder_outputs):
        return self.encoder_trans(encoder_outputs)

    def forward(self, trg_seq, ext_src_seq, init_states, encoder_outputs, encoder_mask, num_oov):
        # trg_seq : [b,t]
        # init_states : [2,b,d]
        # encoder_outputs : [b,t,d]
        # init_states : a tuple of [2, b, d]
        self.lstm.flatten_parameters()
        batch_size, max_len = trg_seq.size()
        hidden_size = encoder_outputs.size(-1)
        memories = self.get_encoder_features(encoder_outputs)
        logits = []
        prev_states = init_states
        prev_context = torch.zeros((batch_size, 1, hidden_size)).cuda()
        # , device=config.device)
        for i in range(max_len):
            y_i = trg_seq[:, i].unsqueeze(1)  # [b, 1]
            embedded = self.embedding(y_i)  # [b, 1, d]
            lstm_inputs = self.reduce_layer(torch.cat([embedded, prev_context], dim=2))
            output, states = self.lstm(lstm_inputs, prev_states)
            # encoder-decoder attention
            context, energy = self.attention(output, memories, encoder_mask)
            concat_input = torch.cat((output, context), dim=2).squeeze(dim=1)
            logit_input = torch.tanh(self.concat_layer(concat_input))
            logit = self.logit_layer(logit_input)  # [b, |V|]

            # maxout pointer network
            if True:
                # num_oov = max(torch.max(ext_src_seq - self.vocab_size + 1), 0)
                zeros = torch.zeros((batch_size, num_oov)).cuda()
                # , device=config.device)
                extended_logit = torch.cat([logit, zeros], dim=1)
                out = torch.zeros_like(extended_logit) - INF
                out, _ = scatter_max(energy, ext_src_seq, out=out)
                out = out.masked_fill(out == -INF, 0)
                logit = extended_logit + out
                logit = logit.masked_fill(logit == 0, -INF)

            logits.append(logit)
            # update prev state and context
            prev_states = states
            prev_context = context

        logits = torch.stack(logits, dim=1)  # [b, t, |V|]
        return logits

    def decode(self, y, ext_x, prev_states, prev_context, encoder_features, encoder_mask):
        # forward one step lstm
        # y : [b]

        embedded = self.embedding(y.unsqueeze(1))
        lstm_inputs = self.reduce_layer(torch.cat([embedded, prev_context], dim=2))
        output, states = self.lstm(lstm_inputs, prev_states)
        context, energy = self.attention(output, encoder_features, encoder_mask)
        concat_input = torch.cat((output, context), dim=2).squeeze(dim=1)
        logit_input = torch.tanh(self.concat_layer(concat_input))
        logit = self.logit_layer(logit_input)  # [b, |V|]

        if True:
            batch_size = y.size(0)
            num_oov = max(torch.max(ext_x - self.vocab_size + 1), 0)
            zeros = torch.zeros((batch_size, num_oov)).cuda()
            # , device=config.device)
            extended_logit = torch.cat([logit, zeros], dim=1)
            out = torch.zeros_like(extended_logit) - INF
            out, _ = scatter_max(energy, ext_x, out=out)
            out = out.masked_fill(out == -INF, 0)
            logit = extended_logit + out
            logit = logit.masked_fill(logit == -INF, 0)
            # forcing UNK prob 0
            logit[:, UNK_ID] = -INF

        return logit, states, context



