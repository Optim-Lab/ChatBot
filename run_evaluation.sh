#!/bin/bash

# Set the GPU index
gpu_idx=0

# Export the environment variable
export CUDA_VISIBLE_DEVICES="${gpu_idx}"

# virutal environment directory
ENV=/home/seunghwan/anaconda3/envs/llm/bin/python

EXECUTION_FILE=/home/seunghwan/ChatBot/6_evaluation.py

experiments=(
"--lora_weights ./models/korani_LORA_000"
# "--lora_weights ./models/korani_LORA_000 --base_model beomi/KoAlpaca-Polyglot-12.8B"
)

for index in ${!experiments[*]}; do
    $ENV $EXECUTION_FILE ${experiments[$index]}
    # sleep 60
done
