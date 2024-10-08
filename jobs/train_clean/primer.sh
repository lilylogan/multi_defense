data=$1
model=$2


job_name=Clean_${data}_${model}

partition_=gpu
mem_=40
time_=1440
constraint_=gpu-40gb


sbatch  --mem=${mem_}G \
        --time=$time_ \
         --partition=$partition_ \
         --constraint=$constraint_ \
         --job-name=$job_name \
         --output=jobs/logs/clean/$job_name \
         --error=jobs/errors/clean/$job_name \
         jobs/train_clean/runner.sh $data $model

