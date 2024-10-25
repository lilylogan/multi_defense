#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --account=uoml
#SBATCH --gpus=1


# module load miniconda
# conda activate wyou-react-20210625-4
# module load java

module load miniconda3/20240410
conda activate badacts_defense
module load java
export PATH="/home/llogan3/.conda/envs/badacts_defense/bin:$PATH"




data=$1
model=$2
filter=$3
poisoner=$4
rs=$5
pr=$6
llm=$7
style=$8



if [[ $poisoner = 'llmbkd' || $poisoner = 'attrbkd' ]]; then

    python3 scripts/demo_attack.py --config_path ./configs/${data}/${model}/attack/${filter}/A_${poisoner}_${llm}_${style}_${pr}_${rs}.json

else
    # python3 scripts/demo_attack.py --config_path ./configs/${data}/${model}/attack/${filter}/A_${poisoner}_${pr}_${rs}.json
    python scripts/demo_attack.py --config_path ./configs/${data}/${model}/defend/${filter}/D_badacts_${poisoner}_${pr}_${rs}.json

fi
