data=$1
llm=$2
poisoner=$3
pr=$4


if [[ $data = 'agnews' ]]; then
    pr_list=("0.002" "0.006" "0.01") #"0.006" "0.008" "0.01" "0.02"
elif [[ $data = 'sst-2' || $data = 'enron' ]]; then
    pr_list=("0.006" "0.008" "0.01" "0.02")
else
    pr_list=()
fi


for pr in ${pr_list[@]}; do
    job_name=Clean_${data}_${llm}_${poisoner}_${pr}

#    partition_=gpu
#    mem_=40
#    time_=1440
#    constraint_=gpu-40gb

    ##echo getting longgpu...
#    partition_=gpulong
    partition_=preempt
    mem_=40
    time_=1440
    constraint_=gpu-40gb


    sbatch  --mem=${mem_}G \
            --time=$time_ \
             --partition=$partition_ \
             --constraint=$constraint_ \
             --job-name=$job_name \
             --output=jobs/logs/clf_clean_poison/$job_name \
             --error=jobs/errors/clf_clean_poison/$job_name \
             jobs/clf_clean_poison/runner.sh $data $llm $poisoner $pr
done

