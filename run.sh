#!/bin/bash

# virutal environment directory
ENV=/home1/seunghwan/miniconda3/envs/llm0/bin/python

# file directory of multiple execution source ".sh"
RUN_SRC=./run_src.sh

# file directory of experiment ".py"
# EXECUTION_FILE=/home1/seunghwan/ChatBot/5_generation.py
EXECUTION_FILE=/home1/seunghwan/ChatBot/6_evaluation.py

# default prefix of job name
DEFAULT_NAME=KorANI

sbatch $RUN_SRC $ENV $EXECUTION_FILE 

