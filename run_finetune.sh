#!/bin/bash

# virutal environment directory
ENV=/home/seunghwan/anaconda3/envs/llm/bin/python

EXECUTION_FILE=/home/seunghwan/ChatBot/4_finetune.py

DATA_ROOT_ARGS=--data-path

# DATA_ROOT=./assets/data/training_data.json
DATA_ROOT=./assets/data/pretrained/training_data.json ### training dataset for pretrained version

GPU_ARGS=--gpu-idx

experiments=(
"--num-epochs 3 --wandb-project huggingface" 
# "--num-epochs 3 --wandb-project huggingface --base-model-name-or-path beomi/KoAlpaca-Polyglot-12.8B"  
)

for index in ${!experiments[*]}; do
    $ENV $EXECUTION_FILE ${experiments[$index]} $DATA_ROOT_ARGS $DATA_ROOT $GPU_ARGS 0
    # sleep 60
done
