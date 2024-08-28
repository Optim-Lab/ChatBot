#!/bin/bash

# virutal environment directory
ENV=~/anaconda3/envs/llm/bin/python

# file directory of experiment ".py"
EXECUTION_FILE=/home/optim1/ChatBot/4_finetune.py

# DATA_ROOT=./assets/data/training_data.json
DATA_ROOT=./assets/data/pretrained/training_data.json ### training dataset for pretrained version

DATA_ROOT_ARGS=--data-path

# python argparse source for experiments
# experiments="--num-epochs 3" ### NO WANDB
experiments="--num-epochs 3 --wandb-project huggingface" ### use WANDB
# experiments=(
# "--num-epochs 3"
# "--num-epochs 5"
# )

for index in ${!experiments[*]}; do
    $ENV $EXECUTION_FILE ${experiments[$index]} $DATA_ROOT_ARGS $DATA_ROOT
    # sleep 60
done
