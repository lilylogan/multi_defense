#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --account=uoml
#SBATCH --gpus=1


module load miniconda
conda activate wyou-react-20210625-4
module load java


data=$1
llm=$2
cluster=$3
cls=$4
style=$5

python3 clustering/cluster_poison_single_label.py \
      --data $data \
      --llm $llm \
      --cluster $cluster \
      --cls $cls \
      --style $style


