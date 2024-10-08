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
poisoner=$3
pr=$4

python3 clf_clean_poison/clf_clean_poison.py \
      --data $data \
      --llm $llm \
      --poisoner $poisoner \
      --poison_rate $pr

