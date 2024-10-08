## clean label visuals

# # SST-2


###
#./jobs/visual/primer.sh 'sst-2' 'roberta' 'filter' 'addsent' '' '' '' ''
#./jobs/visual/primer.sh 'sst-2' 'roberta' 'filter' 'stylebkd' '' '' '' ''
#./jobs/visual/primer.sh 'sst-2' 'roberta' 'filter' 'llmbkd' '' '' 'gpt-3.5-turbo' '' ''
./jobs/visual/primer.sh 'sst-2' 'roberta' 'filter' 'llmbkd' '' '' 'mixtral' '' ''
./jobs/visual/primer.sh 'sst-2' 'roberta' 'filter' 'llmbkd' '' '' 'llama' '' ''
#
#
#
### # Blog

#./jobs/visual/primer.sh 'blog' 'roberta' 'filter' 'addsent' '' '' ''
#./jobs/visual/primer.sh 'blog' 'roberta' 'filter' 'stylebkd' '' '' ''
#./jobs/visual/primer.sh 'blog' 'roberta' 'filter' 'llmbkd' '' '' 'gpt-3.5-turbo' ''
#./jobs/visual/primer.sh 'blog' 'roberta' 'filter' 'attrbkd' '' '' 'mixtral' ''
#./jobs/visual/primer.sh 'blog' 'roberta' 'filter' 'attrbkd' '' '' 'llama' ''
#./jobs/visual/primer.sh 'blog' 'roberta' 'filter' 'attrbkd' '' '' 'gpt-4o' ''
#
#

#
#
## # AG News
###

#./jobs/visual/primer.sh 'agnews' 'roberta' 'filter' 'addsent' '' '' ''
#./jobs/visual/primer.sh 'agnews' 'roberta' 'filter' 'stylebkd' '' '' ''
#./jobs/visual/primer.sh 'agnews' 'roberta' 'filter' 'llmbkd' '' '' 'gpt-3.5-turbo' ''
#

./jobs/visual/primer.sh 'agnews' 'roberta' 'filter' 'llmbkd' '' '' 'gpt-4o' '' ''
./jobs/visual/primer.sh 'agnews' 'roberta' 'filter' 'llmbkd' '' '' 'llama' '' ''


#
#
#
## For efficiency purposes, saving synbkd for the last
#./jobs/visual/primer.sh 'sst-2' 'roberta' 'filter' 'synbkd' '' '' ''
##
#./jobs/visual/primer.sh 'hsol' 'roberta' 'filter' 'synbkd' '' '' ''
##
#./jobs/visual/primer.sh 'toxigen' 'roberta' 'filter' 'synbkd' '' '' ''
#
#./jobs/visual/primer.sh 'agnews' 'roberta' 'filter' 'synbkd' '' '' ''


