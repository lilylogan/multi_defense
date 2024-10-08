llm=$1
data=$2
repetitions=$3
style=$4
clean_split=$5
poison_split=$6
clean_path=$7
poison_path=$8
generated_dir=$9


##echo getting gpu...  #preempt or gpu
partition_=gpu
mem_=40
time_=1440
constraint_=gpu-40gb

##echo getting longgpu...
#partition_=gpulong
#mem_=40
#time_=4320
#constraint_=gpu-40gb


## lowd
#partition_=lowd
#mem_=80
#time_=4320
#constraint_=gpu-80gb,no-mig



job_name=TxtGen_${data}_${llm}_${style}_${repetitions}

#echo $job_name
sbatch  --mem=${mem_}G \
        --time=$time_ \
         --partition=$partition_ \
         --constraint=$constraint_ \
         --job-name=$job_name \
         --output=jobs/output/$job_name \
         --error=jobs/error/$job_name \
         jobs/txt_gen/runner.sh $llm $data $repetitions $style $clean_split $poison_split $clean_path $poison_path $generated_dir

