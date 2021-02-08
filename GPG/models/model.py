import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable
from GPG.models.beamsearch import Beam, sort_beams, Hypothesis
from GPG.data.vocab import Vocab
from GPG.models.model_builder import build_glove_embeddings
from GPG.encoders.graphfusion_encoder import GraphFusionEncoder
from GPG.encoders.graphfusion_encoder_base import GraphFusionEncoder_Base
from GPG.decoders.decoder import Pointer_Decoder, Pointer_Decoder_Base, Decoder


# START_DECODING = '[START]' 
# STOP_DECODING = '[END]' 
# UNKNOWN_TOKEN = '[OOV]' 

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "UNKNOWN"
START_TOKEN = "<s>"
END_TOKEN = "EOS"

PAD_ID = 0
UNK_ID = 1
START_ID = 2
END_ID = 3

class GraphPointerGenerator(nn.Module):
    def __init__(self, config, embeddings = None):
        super(GraphPointerGenerator, self).__init__()
        self.config = config

        # self.encoder = GraphFusionEncoder_Base(self.config, embeddings)
        # self.decoder = Pointer_Decoder_Base(self.config, embeddings)
        self.encoder = GraphFusionEncoder(self.config, embeddings)
        self.decoder = Decoder(self.config, embeddings)

        if config.notrain:
            self.encoder = self.encoder.eval()
            self.decoder = self.decoder.eval()
        else:
            self.encoder = self.encoder.train()
            self.decoder = self.decoder.train()

        self.max_query_length = 50

    def forward(self, batch):
        enc_batch_extend_vocab = None
        coverage = None
        tgt = batch['tgt_idxs']
        enc_padding_mask = batch['context_mask']
        src_seq = batch['context_idxs']

        if self.config.use_copy:
            enc_batch_extend_vocab = batch['context_batch_extend_vocab']
        if self.config.is_coverage:
            coverage = torch.zeros(batch['context_idxs'].size())
        if batch['max_context_oovs'] > 0:
            extra_zeros = torch.zeros(tgt.size(0), batch['max_context_oovs'])
            extra_zeros = extra_zeros.cuda()


        if self.config.use_cuda:
            tgt = tgt.cuda()
            enc_padding_mask = enc_padding_mask.cuda()
            enc_batch_extend_vocab = enc_batch_extend_vocab.cuda()
            coverage = coverage.cuda()
        
        doc_encoder_states, doc_encoder_outputs_update, softmasks = self.encoder(batch)
        
        enc_mask =(src_seq==0).bool() 
        logits = self.decoder(tgt[:, :-1], enc_batch_extend_vocab, doc_encoder_states, doc_encoder_outputs_update, enc_mask, batch['max_context_oovs'])
        return logits, softmasks

        # # outputs, _, attns, p_gens, coverages = self.decoder(tgt[:, :-1], doc_encoder_states, doc_encoder_outputs_update, enc_padding_mask, extra_zeros,enc_batch_extend_vocab, coverage)
        # return outputs, attns, coverages, softmasks
    @staticmethod
    def sort_hypotheses(hypotheses):
        return sorted(hypotheses, key=lambda h: h.avg_log_prob, reverse=True)

    # this if the beamsearch method for the Decoder
    def beam_search(self, batch):
        src_seq = batch['context_idxs']
        # zeros = torch.zeros_like(src_seq)
        # enc_mask = torch.ByteTensor(src_seq == zeros)
        enc_mask =(src_seq==0).bool() 
        prev_context = torch.zeros(1, 1, 2 * self.config.hidden_dim)
        ext_src_seq = batch['context_batch_extend_vocab']

        if self.config.use_cuda:
            ext_src_seq = ext_src_seq.cuda()
            enc_mask = enc_mask.cuda()
            prev_context = prev_context.cuda()

        # enc_outputs, enc_states = self.model.encoder(src_seq, src_len, tag_seq)
        enc_states, enc_outputs, _ = self.encoder(batch)

        h, c = enc_states  # [2, b, d] but b = 1
        hypotheses = [Hypothesis(tokens=[START_ID],
                                 log_probs=[0.0],
                                 state=(h[:, 0, :], c[:, 0, :]),
                                 context=prev_context[0]) for _ in range(self.config.beam_size)]
        ext_src_seq = ext_src_seq.repeat(self.config.beam_size, 1)
        enc_outputs = enc_outputs.repeat(self.config.beam_size, 1, 1)
        enc_features = self.decoder.get_encoder_features(enc_outputs)
        enc_mask = enc_mask.repeat(self.config.beam_size, 1)
        num_steps = 0
        results = []
        while num_steps < self.config.max_tgt_len and len(results) < self.config.beam_size:
            latest_tokens = [h.latest_token for h in hypotheses]
            latest_tokens = [idx if idx < self.config.vocab_size else UNK_ID for idx in latest_tokens]
            prev_y = torch.LongTensor(latest_tokens).view(-1)

            if self.config.use_cuda:
                prev_y = prev_y.cuda()
            # make batch of which size is beam size
            all_state_h = []
            all_state_c = []
            all_context = []
            for h in hypotheses:
                state_h, state_c = h.state  # [num_layers, d]
                all_state_h.append(state_h)
                all_state_c.append(state_c)
                all_context.append(h.context)

            prev_h = torch.stack(all_state_h, dim=1)  # [num_layers, beam, d]
            prev_c = torch.stack(all_state_c, dim=1)  # [num_layers, beam, d]
            prev_context = torch.stack(all_context, dim=0)
            prev_states = (prev_h, prev_c)
            # [beam_size, |V|]
            logits, states, context_vector = self.decoder.decode(prev_y, ext_src_seq,
                                                                       prev_states, prev_context,
                                                                       enc_features, enc_mask)
            h_state, c_state = states
            log_probs = F.log_softmax(logits, dim=1)
            top_k_log_probs, top_k_ids \
                = torch.topk(log_probs, self.config.beam_size * 2, dim=-1)

            all_hypotheses = []
            num_orig_hypotheses = 1 if num_steps == 0 else len(hypotheses)
            for i in range(num_orig_hypotheses):
                h = hypotheses[i]
                state_i = (h_state[:, i, :], c_state[:, i, :])
                context_i = context_vector[i]
                for j in range(self.config.beam_size * 2):
                    new_h = h.extend(token=top_k_ids[i][j].item(),
                                     log_prob=top_k_log_probs[i][j].item(),
                                     state=state_i,
                                     context=context_i)
                    all_hypotheses.append(new_h)

            hypotheses = []
            for h in self.sort_hypotheses(all_hypotheses):
                if h.latest_token == END_ID:
                    if num_steps >= self.config.min_tgt_len:
                        results.append(h)
                else:
                    hypotheses.append(h)

                if len(hypotheses) == self.config.beam_size or len(results) == self.config.beam_size:
                    break
            num_steps += 1
        if len(results) == 0:
            results = hypotheses
        h_sorted = self.sort_hypotheses(results)

        return h_sorted[0]
        
    # this is the beamsearch method for Pointer_Decoder
    def beam_sample(self, batch):
        #batch should have only one example
        # n_gpu = torch.cuda.device_count()
        doc_encoder_states, doc_encoder_outputs_update, _ = self.encoder(batch)
        extra_zeros = None
        enc_batch_extend_vocab = None
        dec_h, dec_c = doc_encoder_states   # 2 x b x hidden_dim
        # dec_h = dec_h.squeeze() # 2 x hidden_dim
        # dec_c = dec_c.squeeze()
        tgt = batch['tgt_idxs']
        enc_padding_mask = batch['context_mask']
        if self.config.use_copy:
            enc_batch_extend_vocab = batch['context_batch_extend_vocab']
        c_t_0 = torch.zeros(tgt.size(0), 2 * self.config.hidden_dim)
        coverage_t_0 = torch.zeros(batch['context_idxs'].size())
        if batch['max_context_oovs'] > 0:
            extra_zeros = torch.zeros((tgt.size(0), batch['max_context_oovs']))
            extra_zeros = extra_zeros.cuda()

        if self.config.use_cuda:
            tgt = tgt.cuda()
            enc_padding_mask = enc_padding_mask.cuda()
            enc_batch_extend_vocab = enc_batch_extend_vocab.cuda()
            c_t_0 = c_t_0.cuda()
            coverage_t_0 = coverage_t_0.cuda()

        #decoder batch preparation, it has beam_size example initially everything is repeated
        beams = [Beam(tokens=[START_ID],
                      log_probs=[0.0],
                      state=(dec_h[:, 0, :], dec_c[:, 0, :]), # 2 x hidden_dim 
                      context = c_t_0[0],
                      coverage=(coverage_t_0[0] if self.config.is_coverage else None))
                 for _ in range(self.config.beam_size)]
        results = []
        steps = 0
        while steps < self.config.max_tgt_len and len(results) < self.config.beam_size:
            latest_tokens = [h.latest_token for h in beams]
            latest_tokens = [t if t < self.config.vocab_size else UNK_ID \
                             for t in latest_tokens]
            y_t_1 = Variable(torch.LongTensor(latest_tokens))
            y_t_1 = y_t_1.cuda()
            all_state_h =[]
            all_state_c = []
            all_context = []

            for h in beams:
                state_h, state_c = h.state # (2 x hidden_dim , 2 x hidden_dim)
                all_state_h.append(state_h) # beam_size * 2 x hidden_size
                all_state_c.append(state_c)
                all_context.append(h.context)

            # s_t_1 = (torch.stack(all_state_h, 0).unsqueeze(0), torch.stack(all_state_c, 0).unsqueeze(0))   # 1 x beam_size x 1*hidden 
            s_t_1 = (torch.stack(all_state_h, dim = 1), torch.stack(all_state_c, dim = 1))   # 2 x beam_size x hidden             
            c_t_1 = torch.stack(all_context, 0)
            coverage_t_1 = None
            if self.config.is_coverage:
                all_coverage = []
                for h in beams:
                    all_coverage.append(h.coverage)
                coverage_t_1 = torch.stack(all_coverage, 0)

            final_dist, s_t, attn_dist, p_gen, coverage_t, c_t = self.decoder.sample_one(y_t_1, s_t_1, doc_encoder_outputs_update, enc_padding_mask, c_t_1,
                                                               extra_zeros, enc_batch_extend_vocab, coverage_t_1)

            log_probs = torch.log(final_dist)
            topk_log_probs, topk_ids = torch.topk(log_probs, self.config.beam_size * 2)

            dec_h, dec_c = s_t  # 2 x b x hidden_size

            all_beams = []
            num_orig_beams = 1 if steps == 0 else len(beams)
            for i in range(num_orig_beams):
                h = beams[i]
                state_i = (dec_h[:, i, :], dec_c[:, i, :])
                context_i = c_t[i]
                coverage_i = (coverage_t[i] if self.config.is_coverage else None)

                for j in range(self.config.beam_size * 2):  # for each of the top 2*beam_size hyps:
                    new_beam = h.extend(token=topk_ids[i, j].item(),
                                   log_prob=topk_log_probs[i, j].item(),
                                   state=state_i,
                                   context=context_i,
                                   coverage=coverage_i)
                    all_beams.append(new_beam)

            beams = []
            for h in sort_beams(beams = all_beams):
                if h.latest_token == END_ID:
                    if steps >= self.config.min_tgt_len:
                        results.append(h)
                else:
                    beams.append(h)
                if len(beams) == self.config.beam_size or len(results) == self.config.beam_size:
                    break
            steps += 1

        if len(results) == 0:
            results = beams

        beams_sorted = sort_beams(beams = results)
        return beams_sorted[0]
