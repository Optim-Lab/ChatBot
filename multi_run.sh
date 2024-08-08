#!/bin/bash

# virutal environment directory
ENV=/home1/prof/jeon/anaconda3/envs/lm/bin/python

# file directory of multiple execution source ".sh"
RUN_SRC=./run_src.sh

# file directory of experiment ".py"
EXECUTION_FILE=/home1/prof/jeon/KorANI/finetune.py

# DATA_ROOT=./assets/data/v.1.2.5/initial_data_type1.json
DATA_ROOT=(
"./assets/data/v.1.2.12/tox_initial_data.json"
)

DATA_ROOT_ARGS=--data-path

# default prefix of job name
DEFAULT_NAME=KorANI

# python argparse source for experiments
experiments="--num-epochs 3"
# experiments=(
# "--num-epochs 3"
# "--num-epochs 5"
# )

for index in ${!DATA_ROOT[*]}; do
    sbatch --job-name=$DEFAULT_NAME$index $RUN_SRC $ENV $EXECUTION_FILE $experiments $DATA_ROOT_ARGS ${DATA_ROOT[$index]}
    # echo --job-name=$DEFAULT_NAME$index $RUN_SRC $ENV $EXECUTION_FILE $experiments $DATA_ROOT_ARGS ${DATA_ROOT[$index]}
    sleep 60
done
# for index in ${!experiments[*]}; do
#     sbatch --job-name=$DEFAULT_NAME$index $RUN_SRC $ENV $EXECUTION_FILE ${experiments[$index]} $DATA_ROOT_ARGS $DATA_ROOT
#     sleep 30
# done
