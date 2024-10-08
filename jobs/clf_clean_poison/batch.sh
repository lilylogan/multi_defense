## OpenBackdoor

# clean model training

#./jobs/train_clean/primer.sh 'sst-2' 'roberta'
#./jobs/train_clean/primer.sh 'hsol' 'roberta'
#./jobs/train_clean/primer.sh 'agnews' 'roberta'
#./jobs/train_clean/primer.sh 'toxigen' 'roberta'
#./jobs/train_clean/primer.sh 'yelp' 'roberta'
#./jobs/clf_clean_poison/primer.sh 'agnews' 'gpt-3.5-turbo' 'baselines' ''
#./jobs/clf_clean_poison/primer.sh 'agnews' 'gpt-3.5-turbo' 'llmbkd' ''
./jobs/clf_clean_poison/primer.sh 'sst-2' 'gpt-4-1106-preview' 'llmbkd' ''
#./jobs/clf_clean_poison/primer.sh 'enron' 'gpt-4-1106-preview' 'llmbkd' ''


# all other target models
#./jobs/train_clean/primer.sh 'sst-2' 'bert'
##./jobs/train_clean/primer.sh 'sst-2' 'albert'
##./jobs/train_clean/primer.sh 'sst-2' 'distilbert'
#./jobs/train_clean/primer.sh 'sst-2' 'xlnet'
##
#./jobs/train_clean/primer.sh 'hsol' 'bert'
#./jobs/train_clean/primer.sh 'hsol' 'albert'
#./jobs/train_clean/primer.sh 'hsol' 'distilbert'
#./jobs/train_clean/primer.sh 'hsol' 'xlnet'
#
#./jobs/train_clean/primer.sh 'toxigen' 'bert'
#./jobs/train_clean/primer.sh 'toxigen' 'albert'
#./jobs/train_clean/primer.sh 'toxigen' 'distilbert'
#./jobs/train_clean/primer.sh 'toxigen' 'xlnet'

#./jobs/train_clean/primer.sh 'agnews' 'bert'
#./jobs/train_clean/primer.sh 'agnews' 'albert'
#./jobs/train_clean/primer.sh 'agnews' 'distilbert'
#./jobs/train_clean/primer.sh 'agnews' 'xlnet'


#./jobs/train_clean/primer.sh 'sst-2' 'roberta-large'
#./jobs/train_clean/primer.sh 'hsol' 'roberta-large'
#./jobs/train_clean/primer.sh 'toxigen' 'roberta-large'

#./jobs/train_clean/primer.sh 'sst-2' 'xlnet'
#./jobs/train_clean/primer.sh 'hsol' 'xlnet'
#./jobs/train_clean/primer.sh 'toxigen' 'xlnet'

#./jobs/train_clean/primer.sh 'agnews' 'roberta-large'
#./jobs/train_clean/primer.sh 'agnews' 'xlnet'