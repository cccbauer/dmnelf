#!/bin/bash
#SBATCH --job-name=deeptrain
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=/home/cccbauer/deep_train_%j.out
/home/cccbauer/.conda/envs/r33_fixed/bin/python -u /home/cccbauer/train.py \
    --win-dir /home/cccbauer/deep_win --out /home/cccbauer/deep_train_out --band ""
