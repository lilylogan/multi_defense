data=$1
model=$2
filter=$3
poisoner=$4
rs=$5
pr=$6
llm=$7
style=$8


rs_list=("0" "2" "42") # "0" "1" "2" "10" "42"  "2" "42"
pr_list=("0.05")
styles=("bible" "shakespeare" "tweets" "lawyers")

##echo getting gpu...
partition_=gpu
mem_=40
time_=1440
constraint_=gpu-40gb

#
##echo getting longgpu...
#partition_=gpulong
#mem_=40
#time_=4320
#constraint_=gpu-40gb


#partition_=lowd
#mem_=40
#time_=1440
##constraint_=a100


if [[ $poisoner = 'llmbkd' || $poisoner = 'attrbkd' ]]; then
    for style in ${styles[@]}; do
        for rs in ${rs_list[@]}; do
            for pr in ${pr_list[@]}; do
                job_name=A_${data}_${model}_${filter}_${poisoner}_${llm}_${style}_${pr}_${rs}

                sbatch  --mem=${mem_}G \
                        --time=$time_ \
                         --partition=$partition_ \
                         --constraint=$constraint_ \
                         --job-name=$job_name \
                         --output=jobs/logs/attack/$job_name \
                         --error=jobs/errors/attack/$job_name \
                         jobs/attack/runner.sh $data $model $filter $poisoner $rs $pr $llm $style
            done
        done
    done

else
    for rs in ${rs_list[@]}; do
        for pr in ${pr_list[@]}; do
            job_name=A_${data}_${model}_${filter}_${poisoner}_${pr}_${rs}

            srun    --mem=${mem_}G \
                      --time=$time_ \
                     --partition=$partition_ \
                     --constraint=$constraint_ \
                     --job-name=$job_name \
                     --output=jobs/logs/attack/$job_name \
                     --error=jobs/errors/attack/$job_name \
                     jobs/attack/runner.sh $data $model $filter $poisoner $rs $pr
        done
    done
fi
