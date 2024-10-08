data=$1
llm=$2
cluster=$3
cls=$4
style=$5

style_list=("bible" "default_bt" "shakespeare" "tweets" "attr_2" "attr_3" "attr_4") #"0.006" "0.008" "0.01" "0.02" "bible"

for style in ${style_list[@]}; do
    job_name=Cluster_${data}_${llm}_${style}_${cluster}_${cls}

#    partition_=gpu
#    mem_=40
#    time_=1440
#    constraint_=gpu-40gb

    partition_=gpulong
    mem_=40
    time_=4320
    constraint_=gpu-40gb



  sbatch  --mem=${mem_}G \
          --time=$time_ \
           --partition=$partition_ \
           --constraint=$constraint_ \
           --job-name=$job_name \
           --output=jobs/logs/cluster/$job_name \
           --error=jobs/errors/cluster/$job_name \
           jobs/cluster/runner.sh $data $llm $cluster $cls $style

 done