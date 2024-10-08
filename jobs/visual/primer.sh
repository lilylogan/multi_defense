data=$1
model=$2
filter=$3
poisoner=$4
rs=$5
pr=$6
llm=$7
style=$8
layer=$9


rs_list=("0") # "0" "1" "2" "10" "42"  "2" "42"
layers=(-1 1 3 6 9)


if [[ $data = 'sst-2' && $llm = 'llama' ]]; then
    pr_list=("0.05")
    styles=("llm_tweets" "llm_bible")
elif [[ $data = 'sst-2' && $llm = 'mixtral' ]]; then
    pr_list=("0.01")
    styles=("llm_default")
elif [[ $data = 'sst-2' && $llm = 'gpt-3.5-turbo' ]]; then
    pr_list=("0.01")
    styles=("synbkd")
elif [[ $data = 'agnews' && $llm = 'gpt-4o' ]]; then
    pr_list=("0.01")
    styles=("llm_default")
elif [[ $data = 'agnews' && $llm = 'llama' ]]; then
    pr_list=("0.01")
    styles=("fs_3")
elif [[ $data = 'sst-2' && $poisoner = 'addsent' ]]; then
    pr_list=("0.05")
    styles=("")
elif [[ $data = 'sst-2' && $poisoner = 'synbkd' ]]; then
    pr_list=("0.01")
    styles=("")
else
    echo Invalid params
fi



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
                for layer in ${layers[@]}; do
                    job_name=A_${data}_${model}_${filter}_${poisoner}_${llm}_${style}_${pr}_${rs}_${layer}

                    sbatch  --mem=${mem_}G \
                            --time=$time_ \
                             --partition=$partition_ \
                             --constraint=$constraint_ \
                             --job-name=$job_name \
                             --output=jobs/logs/visual/$job_name \
                             --error=jobs/errors/visual/$job_name \
                             jobs/visual/runner.sh $data $model $filter $poisoner $rs $pr $llm $style $layer
                done
            done
        done
    done

else
    for rs in ${rs_list[@]}; do
        for pr in ${pr_list[@]}; do
            for layer in ${layers[@]}; do
                job_name=A_${data}_${model}_${filter}_${poisoner}_${pr}_${rs}_${layer}

                sbatch    --mem=${mem_}G \
                        --time=$time_ \
                         --partition=$partition_ \
                         --constraint=$constraint_ \
                         --job-name=$job_name \
                         --output=jobs/logs/visual/$job_name \
                         --error=jobs/errors/visual/$job_name \
                         jobs/visual/runner.sh $data $model $filter $poisoner $rs $pr $layer
            done
        done
    done
fi
