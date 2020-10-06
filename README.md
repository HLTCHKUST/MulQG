# Multi-hop Question Generation with Graph Convolutional Network (MulQG)
<img src="plot/pytorch-logo-dark.png" width="10%"> [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) 

<img align="right" src="plot/HKUST.jpg" width="12%">

This is the implementation of the paper:

**Multi-hop Question Generation with Graph Convolutional Network**. **Dan Su**, [Yan Xu](https://yana-xuyan.github.io), [Wenlaing Dai](https://wenliangdai.github.io), Ziwei Ji, Tiezheng Yu, Pascale Fung **Findings of EMNLP 2020** [[PDF]](TBC)

If you use any source codes or datasets included in this toolkit in your work, please cite the following paper. The bibtex is listed below:
<pre>
TBC
</pre>

## Abstract

Multi-hop Question Generation (QG) aims to generate answer-related questions by *aggregating* and *reasoning* over multiple scattered evidence from different paragraphs. It is a more challenging yet under-explored task compared to conventional single-hop QG, where the questions are generated from the sentence containing the answer or nearby sentences in the same paragraph without complex reasoning. To address the additional challenges in multi-hop QG, we propose Multi-Hop Encoding Fusion Network for Question Generation (MulQG), which does context encoding in multiple hops with Graph Convolutional Network and encoding fusion via an Encoder Reasoning Gate. To the best of our knowledge, we are the first to tackle the challenge of multi-hop reasoning over paragraphs without any sentence-level information. Empirical results on HotpotQA dataset demonstrate the effectiveness of our method, in comparison with baselines on automatic evaluation metrics. Moreover, from the human evaluation, our proposed model is able to generate fluent questions with high completeness and outperforms the strongest baseline by 20.8% in the multi-hop evaluation.

## Knowledge-embedded Dialogue:
<p align="center">
<img src="plot/main.png" width="100%" />
</p>

Overview of our MulQG framework. In the encoding stage, we pass the initial context encoding \\(C_0\\) and answer encoding \\(A_0\\) to the *Answer-aware Context Encoder* to obtain the first context encoding \\(C_1\\), then \\(C_1\\) and \\(A_0\\) will be used to update a multi-hop answer encoding \\(A_1\\) via the *GCN-based Entity-aware Answer Encoder*, and we use \\(A_1\\) and \\(C_1\\) back to the *Answer-aware Context Encoder* to obtain \\(C_2\\). The final context encoding \\(C_{final}\\) are obtained from the *Encoder Reasoning Gate* which operates over \\(C_1\\) and \\(C_2\\), and will be used in the max-out based decoding stage.

<p align="center">
<img src="plot/graph.png" width="40%" />
</p>

The illustration of GCN-based Entity-aware Answer Encoder.


## Dependencies
You can use `conda` environment yml file (multi-qg.yml) to create your conda environment by running
``` console
❱❱❱ conda env create -f multi-qg.yml
```

## Experiments

* Download the [hotpot QA train and test data](https://github.com/hotpotqa/hotpot) and put them under `./hotpot/data/`.

* Run the preprocessing 

```console
❱❱❱ sh ./run_preprocess.sh
```

* Download the glove embedding and unzip 'glove.840B.300d.txt' and put it under `./glove/glove.840B.300d.txt`

* Run the process_hotpot.py (to obtain the `embedding.pkl` and `word2idx.pkl`)

* Download other pertained models we provided via [link](https://drive.google.com/drive/u/2/folders/167ttUA68L9eVPDni3oh1JIc_28dkAW1P)

Or you can skip the previuos preprocessing step and directly download all the preprocessed files and pre-trained models from the [link](https://drive.google.com/drive/u/2/folders/167ttUA68L9eVPDni3oh1JIc_28dkAW1P)

* Run the training  (Change the configuration file in config.py with proper data path, eg, the log path, the output model path, so on)

```console
❱❱❱ sh ./run_train.sh 
```
