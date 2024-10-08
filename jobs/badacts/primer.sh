##echo getting gpu...
partition_=gpu
mem_=40
time_=1440
constraint_=gpu-40gb

job_name=D_badacts

                    sbatch --mem=${mem_}G \
                             --time=$time_ \
                             --partition=$partition_ \
                             --constraint=$constraint_ \
                             --gres=gpu:1 \
                             --job-name=$job_name \
                             --output=jobs/logs/defend/$job_name \
                             --error=jobs/errors/defend/$job_name \
                             jobs/badacts/runner.sh