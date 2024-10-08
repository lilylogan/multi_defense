#!/bin/bash
#SBATCH --nodes=1
#SBATCH --account=uoml
#SBATCH --gpus=2


# module load mixtral/20240315
module load miniconda3/20240410
conda activate llogan-fabe
export PATH="/home/llogan3/.conda/envs/llogan-fabe/bin:$PATH"
module load java

llm=$1
data=$2
repetitions=$3
style=$4
clean_split=$5
poison_split=$6
clean_path=$7
poison_path=$8
generated_dir=$9

python3 openrouter/call_llm.py \
        --llm $llm \
        --data $data \
        --repetitions $repetitions \
        --style $style \
        --clean_split $clean_split \
        --poison_split $poison_split \
        --clean_path $clean_path \
        --poison_path $poison_path \
        --generated_dir $generated_dir



