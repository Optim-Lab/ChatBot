#!/bin/bash

# virutal environment directory
ENV=/home1/seunghwan/miniconda3/envs/llm/bin/python

# file directory of multiple execution source ".sh"
RUN_SRC=./run_src.sh

# file directory of experiment ".py"
EXECUTION_FILE=/home1/seunghwan/ChatBot/4_finetune.py

# DATA_ROOT=./assets/data/training_data.json
DATA_ROOT=./assets/data/pretrained/training_data.json ### training dataset for pretrained version

DATA_ROOT_ARGS=--data-path

# default prefix of job name
DEFAULT_NAME=KorANI

# python argparse source for experiments
experiments=(
# "--num-epochs 3" ### NO WANDB
"--num-epochs 3 --wandb-project huggingface" ### use WANDB
)

for index in ${!experiments[*]}; do
    sbatch --job-name=$DEFAULT_NAME$index $RUN_SRC $ENV $EXECUTION_FILE ${experiments[$index]} $DATA_ROOT_ARGS $DATA_ROOT
    # sleep 60
done
