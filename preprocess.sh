#!/usr/bin/env bash

# golden paragraphs

WORK_DIR=./hotpot/data_processed
export INPUT_FILE=$1
export OUTPUT_DIR=$2

mkdir ${WORK_DIR}/${OUTPUT_DIR}

python paragraph_selection/select_paras_QG.py \
    --input_path=${INPUT_FILE} \
    --output_path=${WORK_DIR}/${OUTPUT_DIR}/selected_paras_QG.json \
    --ckpt_path=${WORK_DIR}/para_select_model.bin \
    --split=${OUTPUT_DIR}

python bert_ner/predict.py \
    --ckpt_path=work_dir/bert_ner.pt \
    --input_path=${WORK_DIR}/${OUTPUT_DIR}/selected_paras_QG.json \
    --output_path=${WORK_DIR}/${OUTPUT_DIR}/entities.json

python bert_ner/predict_QG.py \
    --use_answer \
    --ckpt_path=work_dir/bert_ner.pt \
    --input_path=${INPUT_FILE} \
    --output_path=${WORK_DIR}/${OUTPUT_DIR}/answer_entities.json

python -m GPG.data.feature \
    --full_data=${INPUT_FILE} \
    --entity_path=${WORK_DIR}/${OUTPUT_DIR}/entities.json \
    --para_path=${WORK_DIR}/${OUTPUT_DIR}/selected_paras_QG.json \
    --example_output=${WORK_DIR}/${OUTPUT_DIR}/examples.pkl.gz \
    --feature_output=${WORK_DIR}/${OUTPUT_DIR}/features.pkl.gz  

python -m GPG.data.create_graph \
    --example_path=${WORK_DIR}/${OUTPUT_DIR}/examples.pkl.gz \
    --feature_path=${WORK_DIR}/${OUTPUT_DIR}/features.pkl.gz \
    --graph_path=${WORK_DIR}/${OUTPUT_DIR}/graph.pkl.gz \
    --query_entity_path=${WORK_DIR}/${OUTPUT_DIR}/answer_entities.json
