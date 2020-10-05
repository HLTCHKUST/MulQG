# MulQG
0. You can use condo environment yml file (multi-qg.yml) to create your conda environment. 

1. Download the hotpot QA train and test data and put them under ./hotpot/data/ https://github.com/hotpotqa/hotpot

2. run the preprocessing 
```console
./run_preprocess.sh
```
3. Download the glove embedding and unzip 'glove.840B.300d.txt' and put it under `./glove/glove.840B.300d.txt`

4. run the process_hotpot.py (to obtain the `embedding.pkl` and `word2idx.pkl`)

5. Download other pertained models we provided via [link](https://drive.google.com/drive/u/2/folders/167ttUA68L9eVPDni3oh1JIc_28dkAW1P)


Or you can skip the previuos preprocessing step and directly download all the preprocessed files and pre-trained models from the [link](https://drive.google.com/drive/u/2/folders/167ttUA68L9eVPDni3oh1JIc_28dkAW1P)

5. run the training  (Change the configuration file in config.py with proper data path, eg, the log path, the output model path, so on)
```console
./run_train.sh 
```