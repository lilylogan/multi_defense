## Defenses for clean label attacks

# data=$1
# model=$2
# filter=$3
# poisoner=$4
# pr=$5
# defender=$6
# dp_ratio=$7
# rs=$8
# style=$9
# seed=$10
# ./jobs/purify/primer.sh 'data' 'model' 'filter' 'poisoner' 'pr' 'defender' 'dp_ratio' 'rs' 'style' 'seed'

## badacts
./jobs/purify/primer.sh 'sst-2' 'roberta' 'nofilter' 'badnets' '' 'badacts' '' '' '' '2024'
# ./jobs/purify/primer.sh 'sst-2' 'bert' 'nofilter' 'badnets' '' 'badacts' '' '' '' 
# ./jobs/purify/primer.sh 'sst-2' 'bert' 'nofilter' 'badnets' '' 'addsent' '' '' '' 
# ./jobs/purify/primer.sh 'sst-2' 'bert' 'nofilter' 'badnets' '' 'stylebkd' '' '' '' 
# ./jobs/purify/primer.sh 'sst-2' 'bert' 'nofilter' 'badnets' '' 'synbkd' '' '' '' 

# ./jobs/purify/primer.sh 'yelp' 'bert' 'nofilter' 'badnets' '' 'badacts' '' '' '' 
# ./jobs/purify/primer.sh 'yelp' 'bert' 'nofilter' 'badnets' '' 'addsent' '' '' '' 
# ./jobs/purify/primer.sh 'yelp' 'bert' 'nofilter' 'badnets' '' 'stylebkd' '' '' '' 
# ./jobs/purify/primer.sh 'yelp' 'bert' 'nofilter' 'badnets' '' 'synbkd' '' '' '' 

# ./jobs/purify/primer.sh 'hsol' 'bert' 'nofilter' 'badnets' '' 'badacts' '' '' '' 
# ./jobs/purify/primer.sh 'hsol' 'bert' 'nofilter' 'badnets' '' 'addsent' '' '' '' 
# ./jobs/purify/primer.sh 'hsol' 'bert' 'nofilter' 'badnets' '' 'stylebkd' '' '' '' 
# ./jobs/purify/primer.sh 'hsol' 'bert' 'nofilter' 'badnets' '' 'synbkd' '' '' '' 

# ./jobs/purify/primer.sh 'agnews' 'bert' 'nofilter' 'badnets' '' 'badacts' '' '' '' 
# ./jobs/purify/primer.sh 'agnews' 'bert' 'nofilter' 'badnets' '' 'addsent' '' '' '' 
# ./jobs/purify/primer.sh 'agnews' 'bert' 'nofilter' 'badnets' '' 'stylebkd' '' '' '' 
# ./jobs/purify/primer.sh 'agnews' 'bert' 'nofilter' 'badnets' '' 'synbkd' '' '' '' 