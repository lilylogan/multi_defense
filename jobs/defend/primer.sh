data=$1
model=$2
filter=$3
poisoner=$4
pr=$5
defender=$6
dp_ratio=$7
rs=$8
style=$9

rs_list=("0") # "1" "2" "10" "42"
pr_list=("0.05")
dp_ratio_list=("0.1") # "0.1" "0.2" "0.3" "0.4" "0.5" "0.6" "0.7" "0.8"

#if [[ $data = 'sst-2' ]]; then
#    styles=("default" "bible" "Shakespeare" "lawyers" "Gen-z" "sports" "Irish" "Australian" "politicians" "TikTok" "gangster" "rare")
#elif [[ $data = 'hsol' || 'toxigen' || 'agnews' ]]; then
#    styles=("default" "bible" "Gen-z" "sports")
#
#else
#    styles=("")
#fi

# styles=("default" "bible" "Gen-z" "sports")
styles=($style)

##echo getting gpu...
partition_=gpu
mem_=40
time_=1440
constraint_=gpu-40gb



for pr in ${pr_list[@]}; do
    if [[ $defender = 'react'  &&  ( $poisoner = 'llmbkd' || $poisoner = 'attrbkd' ) ]]; then
        echo Scenario one...
        for style in ${styles[@]}; do
            for rs in ${rs_list[@]}; do
                for dp_ratio in ${dp_ratio_list[@]}; do
                    job_name=D_${defender}_${dp_ratio}_${poisoner}_${style}_${pr}_${rs}_${model}

                    sbatch --mem=${mem_}G \
                             --time=$time_ \
                             --partition=$partition_ \
                             --constraint=$constraint_ \
                             --gres=gpu:1 \
                             --job-name=$job_name \
                             --output=jobs/logs/defend/$job_name \
                             --error=jobs/errors/defend/$job_name \
                             jobs/defend/runner.sh $data $model $filter $poisoner $pr $defender ${dp_ratio} ${rs} ${style}

                done
            done
        done

    elif [[ $defender = 'react'  &&  ( $poisoner != 'llmbkd' && $poisoner != 'attrbkd' ) ]]; then
        for rs in ${rs_list[@]}; do
            for dp_ratio in ${dp_ratio_list[@]}; do
                job_name=D_${data}_${defender}_${dp_ratio}_${poisoner}_${pr}_${rs}

                sbatch --mem=${mem_}G \
                         --time=$time_ \
                         --partition=$partition_ \
                         --constraint=$constraint_ \
                         --gres=gpu:1 \
                         --job-name=$job_name \
                         --output=jobs/logs/defend/$job_name \
                         --error=jobs/errors/defend/$job_name \
                         jobs/defend/runner.sh $data $model $filter $poisoner $pr $defender ${dp_ratio} ${rs}
            done
        done
    elif [[ $defender != 'react'  &&  ( $poisoner = 'llmbkd' || $poisoner = 'attrbkd' ) ]]; then
        for style in ${styles[@]}; do
            for rs in ${rs_list[@]}; do
                job_name=D_${data}_${defender}_${poisoner}_${style}_${pr}_${rs}

                sbatch --mem=${mem_}G \
                         --time=$time_ \
                         --partition=$partition_ \
                         --constraint=$constraint_ \
                         --gres=gpu:1 \
                         --job-name=$job_name \
                         --output=jobs/logs/defend/$job_name \
                         --error=jobs/errors/defend/$job_name \
                         jobs/defend/runner.sh $data $model $filter $poisoner $pr $defender "" ${rs} ${style}
            done
        done

    else
        for rs in ${rs_list[@]}; do
            job_name=D_${data}_${defender}_${poisoner}_${pr}_${rs}_${model}

            sbatch --mem=${mem_}G \
                     --time=$time_ \
                     --partition=$partition_ \
                     --constraint=$constraint_ \
                     --gres=gpu:1 \
                     --job-name=$job_name \
                     --output=jobs/logs/defend/$job_name \
                     --error=jobs/errors/defend/$job_name \
                     jobs/defend/runner.sh $data $model $filter $poisoner $pr $defender "" ${rs}
        done
    fi

done


