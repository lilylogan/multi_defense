## clean label attacks

# # SST-2


###
#./jobs/attack/primer.sh 'sst-2' 'roberta' 'filter' 'addsent' '' '' ''
#./jobs/attack/primer.sh 'sst-2' 'roberta' 'filter' 'stylebkd' '' '' ''
#./jobs/attack/primer.sh 'sst-2' 'roberta' 'filter' 'llmbkd' '' '' 'gpt-3.5-turbo' ''
#./jobs/attack/primer.sh 'sst-2' 'roberta' 'filter' 'attrbkd' '' '' 'mixtral' ''
#./jobs/attack/primer.sh 'sst-2' 'roberta' 'filter' 'attrbkd' '' '' 'llama' ''
#
#
#
### # Blog

#./jobs/attack/primer.sh 'blog' 'roberta' 'filter' 'addsent' '' '' ''
#./jobs/attack/primer.sh 'blog' 'roberta' 'filter' 'stylebkd' '' '' ''
#./jobs/attack/primer.sh 'blog' 'roberta' 'filter' 'llmbkd' '' '' 'gpt-3.5-turbo' ''
#./jobs/attack/primer.sh 'blog' 'roberta' 'filter' 'attrbkd' '' '' 'mixtral' ''
#./jobs/attack/primer.sh 'blog' 'roberta' 'filter' 'attrbkd' '' '' 'llama' ''
#./jobs/attack/primer.sh 'blog' 'roberta' 'filter' 'attrbkd' '' '' 'gpt-4o' ''
#
#

#
#
## # AG News
###

#./jobs/attack/primer.sh 'agnews' 'roberta' 'filter' 'addsent' '' '' ''
#./jobs/attack/primer.sh 'agnews' 'roberta' 'filter' 'stylebkd' '' '' ''
#./jobs/attack/primer.sh 'agnews' 'roberta' 'filter' 'llmbkd' '' '' 'gpt-3.5-turbo' ''
#




#
#
#
## For efficiency purposes, saving synbkd for the last
#./jobs/attack/primer.sh 'sst-2' 'roberta' 'filter' 'synbkd' '' '' ''
##
#./jobs/attack/primer.sh 'hsol' 'roberta' 'filter' 'synbkd' '' '' ''
##
#./jobs/attack/primer.sh 'toxigen' 'roberta' 'filter' 'synbkd' '' '' ''
#
#./jobs/attack/primer.sh 'agnews' 'roberta' 'filter' 'synbkd' '' '' ''


# BadActs
# data model filter poisoner rs pr llm style
./jobs/attack/primer.sh 'sst-2' 'roberta' 'nofilter' 'badnets' '' '' '' 