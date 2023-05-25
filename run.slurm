#!/bin/bash
#SBATCH -J PCFG
#SBATCH -p critical_a40
#SBATCH --cpus-per-task=4
#SBATCH -N 1
#SBATCH -t 20:30:30
#SBATCH --gres=gpu:1
#SBATCH --output=negra.out
#SBATCH --error=negra.err

python train.py +exp=













