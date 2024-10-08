
# ./jobs/txt_gen/primer.sh 'llama' 'sst-2' '4' 'llm_bible' '200' '200'
# ./jobs/txt_gen/primer.sh 'llama' 'sst-2' '4' 'llm_default' '200' '200'

# ./jobs/txt_gen/primer.sh 'llama' 'agnews' '4' 'llm_bible' '200' '200'
# ./jobs/txt_gen/primer.sh 'llama' 'agnews' '4' 'llm_default' '200' '200'

### sst-2
#./jobs/txt_gen/primer.sh 'sst-2' 'mixtral' 'g1'
#./jobs/txt_gen/primer.sh 'sst-2' 'mixtral' 'g2'
#./jobs/txt_gen/primer.sh 'sst-2' 'mixtral' 'g3'
#./jobs/txt_gen/primer.sh 'sst-2' 'mixtral' 'g4'
# ./jobs/txt_gen/primer.sh 'llama' 'sst-2' '4' 'none' '200' '200' 'poison_data/baselines/sst-2/1/addsent/filter/train-clean.csv' 'poison_data/baselines/sst-2/1/addsent/filter/train-poison.csv' 'generated_data/sst-2/addsent'
./jobs/txt_gen/primer.sh 'llama' 'sst-2' '4' 'none' '200' '200' 'poison_data/baselines/sst-2/1/synbkd/filter/train-clean.csv' 'poison_data/baselines/sst-2/1/synbkd/filter/train-poison.csv' 'generated_data/sst-2/synbkd'
# ./jobs/txt_gen/primer.sh '' '' '' '' '' '' 'poison_data/baselines/sst-2/1/addsent/filter/train-clean.csv' 'poison_data/baselines/sst-2/1/addsent/train-poison.csv' ''
# ./jobs/txt_gen/primer.sh 'llama' 'agnews' '4' 'none' '200' '200' 'poison_data/baselines/agnews/0/addsent/filter/train-clean.csv' 'poison_data/baselines/agnews/0/addsent/filter/train-poison.csv' 'generated_data/agnews/addsent'
# ./jobs/txt_gen/primer.sh 'llama' 'agnews' '4' 'none' '200' '200' 'poison_data/baselines/agnews/0/synbkd/filter/train-clean.csv' 'poison_data/baselines/agnews/0/addsent/filter/train-poison.csv' 'generated_data/agnews/synbkd'
# ./jobs/txt_gen/primer.sh '' 'agnews' '' '' '' '' 's' 's' ''



#./jobs/txt_gen/primer.sh 'sst-2' 'llama' 'g1'
#./jobs/txt_gen/primer.sh 'sst-2' 'mixtral' 'g1'
#./jobs/txt_gen/primer.sh 'sst-2' 'llama' 'g2'
#./jobs/txt_gen/primer.sh 'sst-2' 'llama' 'g3'
#
#
#./jobs/txt_gen/primer.sh 'sst-2' 'gpt-4o' 'g1'
#
#
#
##### agnews
##./jobs/txt_gen/primer.sh 'agnews' 'mixtral' 'g1'
#./jobs/txt_gen/primer.sh 'agnews' 'llama' 'g1'
#./jobs/txt_gen/primer.sh 'agnews' 'llama' 'g2'
#
#./jobs/txt_gen/primer.sh 'agnews' 'gpt-4o' 'g1'
#
##
#### blog
#./jobs/txt_gen/primer.sh 'blog' 'mixtral' 'g1'
#./jobs/txt_gen/primer.sh 'blog' 'llama' 'g1'
#./jobs/txt_gen/primer.sh 'blog' 'llama' 'g2'
#./jobs/txt_gen/primer.sh 'blog' 'gpt-4o' 'g1'


