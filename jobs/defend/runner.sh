#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --account=uoml
#SBATCH --gpus=1


# module load miniconda
# conda activate wyou-react-20210625-4

# fabe
# module load miniconda3/20240410
# conda activate llogan-fabe
# export PATH="/home/llogan3/.conda/envs/llogan-fabe/bin:$PATH"
# module load java

# badacts
module load miniconda3/20240410
conda activate badacts_defense
module load java
export PATH="/home/llogan3/.conda/envs/badacts_defense/bin:$PATH"



data=$1
model=$2
filter=$3
poisoner=$4
pr=$5
defender=$6
dp_ratio=$7
rs=$8
style=$9

if [[ $defender = 'badacts' ]]; then
    if [[ $poisoner == 'llmbkd' || $poisoner == 'attrbkd' ]]; then
        python3 scripts/BadActs_detection.py \
                --config_path ./configs/${data}/${model}/defend/${filter}/D_${defender}_${poisoner}_llama_${style}_${pr}_${rs}.json \
                # --seed 42 
                # --generated_path ./generated_data/${data}/${style}/generated_combined_data.csv
    else
        python3 scripts/BadActs_detection.py \
                --config_path ./configs/${data}/${model}/defend/${filter}/D_${defender}_${poisoner}_${pr}_${rs}.json \
                # --seed 42 
                # --generated_path ./generated_data/${data}/${poisoner}/generated_combined_data.csv
    fi

elif [[ $defender = 'fabe' ]]; then
    if [[ $poisoner == 'llmbkd' || $poisoner == 'attrbkd' ]]; then
        python3 scripts/FABE_defense.py \
                --config_path ./configs/${data}/${model}/defend/${filter}/D_${defender}_${poisoner}_llama_${style}_${pr}_${rs}.json \
                --seed 42 
                # --generated_path ./generated_data/${data}/${style}/generated_combined_data.csv
    else
        python3 scripts/FABE_defense.py \
                --config_path ./configs/${data}/${model}/defend/${filter}/D_${defender}_${poisoner}_${pr}_${rs}.json \
                --seed 42 
                # --generated_path ./generated_data/${data}/${poisoner}/generated_combined_data.csv
    fi
else
    if [[ $defender = 'react' ]]; then
        if [[ $poisoner = 'llmbkd' || $poisoner = 'attrbkd' ]]; then
            python3 scripts/demo_defend.py --config_path ./configs/${data}/${model}/defend/${filter}/D_${defender}_${dp_ratio}_${poisoner}_${style}_${pr}_${rs}.json 
        else
            python3 scripts/demo_defend.py --config_path ./configs/${data}/${model}/defend/${filter}/D_${defender}_${dp_ratio}_${poisoner}_${pr}_${rs}.json
        fi
    else
        if [[ $poisoner = 'llmbkd' || $poisoner = 'attrbkd' ]]; then
            python3 scripts/demo_defend.py --config_path ./configs/${data}/${model}/defend/${filter}/D_${defender}_${poisoner}_${style}_${pr}_${rs}.json
        else
            python3 scripts/demo_defend.py --config_path ./configs/${data}/${model}/defend/${filter}/D_${defender}_${poisoner}_${pr}_${rs}.json
        fi
    fi
fi

