import argparse
import os
import json
from os.path import join
from GPG.util.utils import read_config


def process_arguments(args):
    args.n_layers = int(args.gnn.split(':')[1].split(',')[0])
    args.n_heads = int(args.gnn.split(':')[1].split(',')[1])
    # args.max_doc_len = 384


def set_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./hotpot/data_processed/")

    # output dir
    parser.add_argument("--output_dir", type=str, default="./prediction/")
    
    # parser.add_argument("--bert_model", type=str, default='bert-base-uncased',
    #                     help='Currently only support bert-base-uncased and bert-large-uncased')

    # learning and log
    parser.add_argument('--ans_update', action='store_true', help='Whether update answer')

    # bi attn
    parser.add_argument("--bi_attn_drop", type=float, default=0.3)

    # reasoning layer
    parser.add_argument("--num_reason_layers", type=int, default=2)
  
    # graph net
    parser.add_argument('--tok2ent', default='mean_max', type=str, help='{mean, mean_max}')
    parser.add_argument('--gnn', default='gat:2,2', type=str, help='gat:n_layer, n_head')
    parser.add_argument("--gnn_drop", type=float, default=0.5)
    parser.add_argument("--gat_attn_drop", type=float, default=0.5)
    parser.add_argument('--q_attn', action='store_true', help='whether use query attention in GAT')
    parser.add_argument("--lstm_drop", type=float, default=0.2)

    
    # lstm
    parser.add_argument("--trunc_norm_init_std", type=float, default=1e-4)
    parser.add_argument("--rand_unif_init_mag", type=float, default=0.02)

    # 
    # parser.add_argument("--epoch", type=float, default=5)
    parser.add_argument("--batch_size", type=int, default=12)
   
    parser.add_argument("--emb_dim", type=int, default=300)
    parser.add_argument("--hidden_dim", type=int, default=300)
    parser.add_argument("--encoder_num_layers", type=int, default=2)
    parser.add_argument("--decoder_num_layers", type=int, default=2)
    parser.add_argument("--num_layers", type=int, default=2)


    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--use_cuda", action='store_true', help='use cuda')
 
    parser.add_argument('--bfs_clf', action='store_true', help='Add BCELoss on bfs mask')
    parser.add_argument('--bfs_lambda', type=float, default=0.5)
    parser.add_argument("--sp_threshold", type=float, default=0.5)

    parser.add_argument('--gpus', default='', type=str,
                        help="Use CUDA on the listed devices.")
    parser.add_argument('--restore',
                        type=str, default=None,
                        help="restore checkpoint")
    parser.add_argument('--seed', type=int, default=1004, #1234
                        help="Random seed")
    parser.add_argument('--verbose', default=False, action='store_true',
                        help="verbose")
    parser.add_argument('--use_copy', default=True, action="store_true",
                        help='whether to use copy mechanism')
    parser.add_argument('--is_coverage', default=True, action="store_true",
                        help='whether to use coverage mechanism')
    parser.add_argument('--notrain', default=False, action='store_true',
                        help="train or not")

    parser.add_argument('-debug', default=False, action="store_true",
                        help='whether to use debug mode')

    # embedding
    parser.add_argument("--embedding", default='./hotpot/embedding.pkl', help='')
    parser.add_argument("--word2idx_file", default='./hotpot/word2idx.pkl', help='')

    # bert-embeddings
    parser.add_argument("--bert_embedding", action='store_true', help='Whether to use bert embeddings or not')
    parser.add_argument("--position_embeddings_flag", action='store_true', help='Whether to use positional embeddings or not')
    parser.add_argument("--bert_input_size", type=int, default=300,
                        help='bert embedding dimension')
    parser.add_argument("--max_position_embeddings", type=int, default=3)
    parser.add_argument("--position_emb_size", type=int, default=3)
    parser.add_argument("--hidden_dropout_prob", type=int, default=0.1)
    parser.add_argument("--layer_norm_eps", type=float, default=1e-12)
    parser.add_argument("--vocab_size", type=int, default=45000)

    #coverage loss
    parser.add_argument("--cov_loss_wt", type=float, default=1.0)
    #beam_search
    parser.add_argument("--max_tgt_len", type=int, default=30)  
    parser.add_argument("--min_tgt_len", type=int, default=8)   
    parser.add_argument('--beam_search', default=False, action='store_true',
                        help="beam_search")
    parser.add_argument("--beam_size", type=int, default=10)   

    parser.add_argument("--eval_interval", type=int, default=2000)
    parser.add_argument("--save_interval", type=int, default=5000)
    parser.add_argument("--epoch", type=int, default=20)
    parser.add_argument("--log", default = "./output/" )

    parser.add_argument("--optim", type=str, default='sgd')
    parser.add_argument("--learning_rate", type=float, default=0.1)   #0.0005 0.0004
    parser.add_argument("--max_grad_norm", type=int, default=5.0)   # 8.0
    parser.add_argument("--learning_rate_decay", type=float, default=0.5)
    parser.add_argument("--start_decay_at", type=int, default=5) #2
    parser.add_argument("--schedule", action='store_true', help = '')
    

    args = parser.parse_args()
    process_arguments(args)

    return args