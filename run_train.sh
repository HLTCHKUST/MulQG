# train the model
nohup python3 -m GPG.main --use_cuda --schedule --ans_update --q_attn --is_coverage --use_copy --batch_size=36 --beam_search --gpus=0,1,2 --position_embeddings_flag > train_20200522.log &

