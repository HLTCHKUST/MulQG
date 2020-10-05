nohup sh preprocess.sh ./hotpot/data/hotpot_dev_distractor_v1.json dev > preprocess_dev.log &
nohup sh preprocess.sh ./hotpot/data/hotpot_train_v1.1.json train > preprocess_train.log &