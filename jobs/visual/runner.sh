#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --account=uoml
#SBATCH --gpus=1


module load miniconda
conda activate wyou-react-20210625-4
module load java

data=$1
model=$2
filter=$3
poisoner=$4
rs=$5
pr=$6

if [[ $poisoner = 'llmbkd' || $poisoner = 'attrbkd' ]]; then
    llm=$7
    style=$8
    layer=$9
    python3 scripts/demo_attack.py --config_path ./configs/${data}/${model}/visual/${filter}/A_${poisoner}_${llm}_${style}_${pr}_${rs}_${layer}.json
else
    layer=$7
    python3 scripts/demo_attack.py --config_path ./configs/${data}/${model}/visual/${filter}/A_${poisoner}_${pr}_${rs}_${layer}.json
fi
