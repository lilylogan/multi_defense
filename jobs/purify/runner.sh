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
seed=$((10))


#BadActs
if [[ $defender = 'badacts' ]]; then
    if [[ $model = 'bert' ]]; then
        if [[ $poisoner == 'llmbkd' || $poisoner == 'attrbkd' ]]; then
            python3 scripts/BadActs_purification.py \
                    --config_path ./configs/${data}/${model}/defend/${filter}/D_${defender}_${poisoner}_llama_${style}_${pr}_${rs}.json \
                    --seed ${seed}
        else
            python3 scripts/BadActs_purification.py \
                    --config_path ./configs/${data}/${model}/defend/${filter}/D_${defender}_${poisoner}_${pr}_${rs}.json \
                    --seed ${seed}
        fi
    else
        if [[ $poisoner == 'llmbkd' || $poisoner == 'attrbkd' ]]; then
            python3 scripts/purify_roberta.py \
                    --config_path ./configs/${data}/${model}/defend/${filter}/D_${defender}_${poisoner}_llama_${style}_${pr}_${rs}.json \
                    --seed ${seed}
        else
            python3 scripts/purify_roberta.py \
                    --config_path ./configs/${data}/${model}/defend/${filter}/D_${defender}_${poisoner}_${pr}_${rs}.json \
                    --seed ${seed}
        fi
    fi

# FABE
elif [[ $defender = 'fabe' ]]; then
    if [[ $model = 'bert' ]]; then
        if [[ $poisoner == 'llmbkd' || $poisoner == 'attrbkd' ]]; then
            python3 scripts/BadActs_purification.py \
                    --config_path ./configs/${data}/${model}/defend/${filter}/D_${defender}_${poisoner}_llama_${style}_${pr}_${rs}.json \
                    --seed ${seed} 
        else
            python3 scripts/BadActs_purification.py \
                    --config_path ./configs/${data}/${model}/defend/${filter}/D_${defender}_${poisoner}_${pr}_${rs}.json \
                    --seed ${seed} 
        fi
    else
        if [[ $poisoner == 'llmbkd' || $poisoner == 'attrbkd' ]]; then
            python3 scripts/purify_roberta.py \
                    --config_path ./configs/${data}/${model}/defend/${filter}/D_${defender}_${poisoner}_llama_${style}_${pr}_${rs}.json \
                    --seed ${seed}
        else
            python3 scripts/purify_roberta.py \
                    --config_path ./configs/${data}/${model}/defend/${filter}/D_${defender}_${poisoner}_${pr}_${rs}.json \
                    --seed ${seed}
        fi
    fi

# all others
else
    if [[ $model = 'bert' ]]; then
        if [[ $defender = 'react' ]]; then
            if [[ $poisoner = 'llmbkd' || $poisoner = 'attrbkd' ]]; then
                python3 scripts/BadActs_purification.py \
                --config_path ./configs/${data}/${model}/defend/${filter}/D_${defender}_${dp_ratio}_${poisoner}_${style}_${pr}_${rs}.json  \
                --seed ${seed}
            else
                python3 scripts/BadActs_purification.py \
                --config_path ./configs/${data}/${model}/defend/${filter}/D_${defender}_${dp_ratio}_${poisoner}_${pr}_${rs}.json \
                --seed ${seed}
            fi
        else
            if [[ $poisoner = 'llmbkd' || $poisoner = 'attrbkd' ]]; then
                python3 scripts/BadActs_purification.py \
                --config_path ./configs/${data}/${model}/defend/${filter}/D_${defender}_${poisoner}_${style}_${pr}_${rs}.json \
                --seed ${seed}
            else
                python3 scripts/BadActs_purification.py \
                --config_path ./configs/${data}/${model}/defend/${filter}/D_${defender}_${poisoner}_${pr}_${rs}.json \
                --seed ${seed}
            fi
        fi
    else
        if [[ $defender = 'react' ]]; then
            if [[ $poisoner = 'llmbkd' || $poisoner = 'attrbkd' ]]; then
                python3 scripts/purify_roberta.py \
                --config_path ./configs/${data}/${model}/defend/${filter}/D_${defender}_${dp_ratio}_${poisoner}_${style}_${pr}_${rs}.json \
                --seed ${seed}
            else
                python3 scripts/purify_roberta.py \
                --config_path ./configs/${data}/${model}/defend/${filter}/D_${defender}_${dp_ratio}_${poisoner}_${pr}_${rs}.json\
                --seed ${seed}
            fi
        else
            if [[ $poisoner = 'llmbkd' || $poisoner = 'attrbkd' ]]; then
                python3 scripts/purify_roberta.py \
                --config_path ./configs/${data}/${model}/defend/${filter}/D_${defender}_${poisoner}_${style}_${pr}_${rs}.json \
                --seed ${seed}
            else
                python3 scripts/purify_roberta.py \
                --config_path ./configs/${data}/${model}/defend/${filter}/D_${defender}_${poisoner}_${pr}_${rs}.json \
                --seed ${seed}
            fi
        fi
    fi
fi

