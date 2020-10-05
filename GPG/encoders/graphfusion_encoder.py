from GPG.encoders.rnn_encoder import RNNEncoder
from GPG.encoders.layers import ReasonLayer, AnsUpdateLayer, ReduceState, Reason_Ans_InterBlocks
from GPG.models.model_utils import cal_position_id_2
import torch
from torch import nn
from GPG.models.model_utils import init_linear_wt, init_lstm_wt



class GraphFusionEncoder(nn.Module):
    def __init__(self, config, embeddings = None):
        super(GraphFusionEncoder, self).__init__()

        self.config = config

        self.word_embeddings = nn.Embedding(config.vocab_size, config.emb_dim)
        if config.position_embeddings_flag:
            self.position_embeddings = nn.Embedding(3, 3)
            emb_dim = config.emb_dim + config.position_emb_size
        else:
            emb_dim = config.emb_dim

        if embeddings is not None:
            self.word_embeddings = nn.Embedding(config.vocab_size, config.emb_dim). \
                from_pretrained(embeddings, freeze=True)

        emb_dim_ans = self.config.emb_dim

        self.num_reason_layers = self.config.num_reason_layers

        self.passage_encoder=nn.LSTM(emb_dim, self.config.hidden_dim, num_layers=self.config.encoder_num_layers, dropout = self.config.dropout, batch_first=True, bidirectional=True)
        self.ans_encoder = nn.LSTM(emb_dim_ans, self.config.hidden_dim, num_layers=self.config.encoder_num_layers, dropout = self.config.dropout, batch_first=True, bidirectional=True)
        init_lstm_wt(self.passage_encoder)
        init_lstm_wt(self.ans_encoder)
        # self.reduce_state = ReduceState(self.config)
        self.reason_ans_inter_blocks = Reason_Ans_InterBlocks(self.config)

    def forward(self, batch):
        
        answer_mapping = batch['ans_mask']
        entity_mask = batch['entity_mask']
        context_ids = batch['context_idxs']
        answer_ids = batch['ans_idxs']
        context_mask = batch['context_mask']
        position_ids = cal_position_id_2(context_ids, batch['y1'], batch['y2'], context_mask)
        src_len = batch['context_lens']

        if self.config.use_cuda:
            answer_mapping = answer_mapping.cuda()
            entity_mask = entity_mask.cuda()
            context_ids = context_ids.cuda()
            answer_ids = answer_ids.cuda()
            position_ids = position_ids.cuda()
            self.word_embeddings = self.word_embeddings.cuda()
            if self.position_embeddings is not None:
                self.position_embeddings = self.position_embeddings.cuda()

        embedded = self.word_embeddings(context_ids)
        if self.position_embeddings is not None:
            position_embedded = self.position_embeddings(position_ids)
            embedded = torch.cat((embedded, position_embedded), dim=2)
        ans_embedded = self.word_embeddings(answer_ids)

        self.passage_encoder.flatten_parameters()
        self.ans_encoder.flatten_parameters()

        doc_encoder_outputs, doc_encoder_hidden = self.passage_encoder(embedded)  # batch x seq x 2*hidden
        ans_encoder_outputs, ans_encoder_hidden = self.ans_encoder(ans_embedded)

        doc_encoder_outputs_update, softmasks = self.reason_ans_inter_blocks(doc_encoder_outputs, ans_encoder_outputs, batch)
        # doc_encoder_states = self.reduce_state(doc_encoder_hidden)
        
        h, c = doc_encoder_hidden # 2*2 x batch_size x hidden_dim
        _, b, d = h.size()
        h = h.view(2, 2, b, d)  # [n_layers, bi, b, d]
        h = torch.cat((h[:, 0, :, :], h[:, 1, :, :]), dim=-1)  # 2 x b x 2*hidden_dim
        # h = h[:, 0, :, :]
        c = c.view(2, 2, b, d)
        # c = c[:, 0, :, :]
        c = torch.cat((c[:, 0, :, :], c[:, 1, :, :]), dim=-1)
        doc_encoder_states = (h, c) 
             
        return doc_encoder_states, doc_encoder_outputs_update, softmasks
    


