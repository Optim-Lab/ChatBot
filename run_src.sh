#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gpu1
#SBATCH --gres=gpu:1

#SBATCH --job-name=experiment
#SBATCH -o logs/s_%j.out
#SBATCH -e logs/s_%j.err

hostname

for argv in "$*"
do
    $argv
done
