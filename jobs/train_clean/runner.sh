#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --account=uoml
#SBATCH --gpus=1

# module load miniconda
# conda activate wyou-react-20210625-4
# module load java
# module load miniconda3/20240410
# conda activate llogan-fabe
# export PATH="/home/llogan3/.conda/envs/llogan-fabe/bin:$PATH"
# module load java
module load miniconda3/20240410
conda activate badacts_defense
module load java
export PATH="/home/llogan3/.conda/envs/badacts_defense/bin:$PATH"


data=$1
model=$2


python3 scripts/train_clean.py \
      --data $data \
      --model $model

