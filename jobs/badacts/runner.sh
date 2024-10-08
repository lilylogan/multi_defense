#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --account=uoml
#SBATCH --gpus=1


# module load miniconda
# conda activate wyou-react-20210625-4
module load miniconda3/20240410
conda activate badacts_defense
export PATH="/home/llogan3/.conda/envs/badacts_defense/bin:$PATH"
module load java

python3 scripts/BadActs_detection.py