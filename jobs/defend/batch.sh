## Defenses for clean label attacks

## badacts
# ./jobs/defend/primer.sh 'sst-2' 'bert' 'nofilter' 'badnets' '' 'badacts' '' '' '' 
# ./jobs/defend/primer.sh 'sst-2' 'bert' 'nofilter' 'badnets' '' 'addsent' '' '' '' 
# ./jobs/defend/primer.sh 'sst-2' 'bert' 'nofilter' 'badnets' '' 'stylebkd' '' '' '' 
# ./jobs/defend/primer.sh 'sst-2' 'bert' 'nofilter' 'badnets' '' 'synbkd' '' '' '' 

# ./jobs/defend/primer.sh 'yelp' 'bert' 'nofilter' 'badnets' '' 'badacts' '' '' '' 
# ./jobs/defend/primer.sh 'yelp' 'bert' 'nofilter' 'badnets' '' 'addsent' '' '' '' 
# ./jobs/defend/primer.sh 'yelp' 'bert' 'nofilter' 'badnets' '' 'stylebkd' '' '' '' 
# ./jobs/defend/primer.sh 'yelp' 'bert' 'nofilter' 'badnets' '' 'synbkd' '' '' '' 

# ./jobs/defend/primer.sh 'hsol' 'bert' 'nofilter' 'badnets' '' 'badacts' '' '' '' 
# ./jobs/defend/primer.sh 'hsol' 'bert' 'nofilter' 'badnets' '' 'addsent' '' '' '' 
# ./jobs/defend/primer.sh 'hsol' 'bert' 'nofilter' 'badnets' '' 'stylebkd' '' '' '' 
# ./jobs/defend/primer.sh 'hsol' 'bert' 'nofilter' 'badnets' '' 'synbkd' '' '' '' 

# ./jobs/defend/primer.sh 'agnews' 'bert' 'nofilter' 'badnets' '' 'badacts' '' '' '' 
# ./jobs/defend/primer.sh 'agnews' 'bert' 'nofilter' 'badnets' '' 'addsent' '' '' '' 
# ./jobs/defend/primer.sh 'agnews' 'bert' 'nofilter' 'badnets' '' 'stylebkd' '' '' '' 
# ./jobs/defend/primer.sh 'agnews' 'bert' 'nofilter' 'badnets' '' 'synbkd' '' '' '' 


## fabe with fine tuning
# ./jobs/defend/primer.sh 'hsol' 'bert' 'nofilter' 'addsent' '' 'fabe' '' '' ''
./jobs/defend/primer.sh 'hsol' 'bert' 'nofilter' 'synbkd' '' 'fabe' '' '' ''

# ./jobs/defend/primer.sh 'sst-2' 'bert' 'nofilter' 'badnets' '' 'fabe' '' '' ''
./jobs/defend/primer.sh 'sst-2' 'bert' 'nofilter' 'synbkd' '' 'fabe' '' '' ''

# ./jobs/defend/primer.sh 'offenseval' 'bert' 'nofilter' 'badnets' '' 'fabe' '' '' ''
./jobs/defend/primer.sh 'offenseval' 'bert' 'nofilter' 'synbkd' '' 'fabe' '' '' ''



## modified fabe
# ./jobs/defend/primer.sh 'sst-2' 'roberta' 'filter' 'attrbkd' '' 'fabe' '' '' 'llm_bible' 
# ./jobs/defend/primer.sh 'sst-2' 'roberta' 'filter' 'attrbkd' '' 'fabe' '' '' 'llm_default'
# ./jobs/defend/primer.sh 'agnews' 'roberta' 'filter' 'attrbkd' '' 'fabe' '' '' 'llm_bible' 
# ./jobs/defend/primer.sh 'agnews' 'roberta' 'filter' 'attrbkd' '' 'fabe' '' '' 'llm_default' 

# ./jobs/defend/primer.sh 'sst-2' 'bert' 'filter' 'addsent' '' 'fabe' '' '' '' 
# ./jobs/defend/primer.sh 'sst-2' 'bert' 'filter' 'synbkd' '' 'fabe' '' '' ''
# ./jobs/defend/primer.sh 'agnews' 'bert' 'filter' 'addsent' '' 'fabe' '' '' ''
# ./jobs/defend/primer.sh 'agnews' 'bert' 'filter' 'synbkd' '' 'fabe' '' '' ''
 

# ./jobs/defend/primer.sh 'sst-2' 'roberta' 'filter' 'addsent' '' 'fabe' '' '' '' 
# ./jobs/defend/primer.sh 'sst-2' 'roberta' 'filter' 'synbkd' '' 'fabe' '' '' ''
# ./jobs/defend/primer.sh 'agnews' 'roberta' 'filter' 'addsent' '' 'fabe' '' '' ''
# ./jobs/defend/primer.sh 'agnews' 'roberta' 'filter' 'synbkd' '' 'fabe' '' '' ''
 


## sst-2
## react defense
#./jobs/defend/primer.sh 'sst-2' 'roberta' 'filter' 'addsent' '' 'react' '' '' ''
#./jobs/defend/primer.sh 'sst-2' 'roberta' 'filter' 'llmbkd' '' 'react' '' '' ''
#./jobs/defend/primer.sh 'sst-2' 'roberta' 'filter' 'stylebkd' '' 'react' '' '' ''
##./jobs/defend/primer.sh 'sst-2' 'roberta' 'filter' 'synbkd' '' 'react' '' '' ''


### other defenses
#### addsent
# ./jobs/defend/primer.sh 'sst-2' 'roberta' 'filter' 'addsent' '' 'bki' '' '' ''
##./jobs/defend/primer.sh 'sst-2' 'roberta' 'filter' 'addsent' '' 'cube' '' '' ''
##./jobs/defend/primer.sh 'sst-2' 'roberta' 'filter' 'addsent' '' 'onion' '' '' ''
##./jobs/defend/primer.sh 'sst-2' 'roberta' 'filter' 'addsent' '' 'rap' '' '' ''
##./jobs/defend/primer.sh 'sst-2' 'roberta' 'filter' 'addsent' '' 'strip' '' '' ''

### stylebkd
#./jobs/defend/primer.sh 'sst-2' 'roberta' 'filter' 'stylebkd' '' 'bki' '' '' ''
#./jobs/defend/primer.sh 'sst-2' 'roberta' 'filter' 'stylebkd' '' 'cube' '' '' ''
#./jobs/defend/primer.sh 'sst-2' 'roberta' 'filter' 'stylebkd' '' 'onion' '' '' ''
#./jobs/defend/primer.sh 'sst-2' 'roberta' 'filter' 'stylebkd' '' 'rap' '' '' ''
#./jobs/defend/primer.sh 'sst-2' 'roberta' 'filter' 'stylebkd' '' 'strip' '' '' ''
##
### synbkd
#./jobs/defend/primer.sh 'sst-2' 'roberta' 'filter' 'synbkd' '' 'bki' '' '' ''
#./jobs/defend/primer.sh 'sst-2' 'roberta' 'filter' 'synbkd' '' 'cube' '' '' ''
#./jobs/defend/primer.sh 'sst-2' 'roberta' 'filter' 'synbkd' '' 'onion' '' '' ''
#./jobs/defend/primer.sh 'sst-2' 'roberta' 'filter' 'synbkd' '' 'rap' '' '' ''
#./jobs/defend/primer.sh 'sst-2' 'roberta' 'filter' 'synbkd' '' 'strip' '' '' ''
#
## llmbkd
#./jobs/defend/primer.sh 'sst-2' 'roberta' 'filter' 'llmbkd' '' 'bki' '' '' ''
#./jobs/defend/primer.sh 'sst-2' 'roberta' 'filter' 'llmbkd' '' 'cube' '' '' ''
#./jobs/defend/primer.sh 'sst-2' 'roberta' 'filter' 'llmbkd' '' 'onion' '' '' ''
#./jobs/defend/primer.sh 'sst-2' 'roberta' 'filter' 'llmbkd' '' 'rap' '' '' ''
#./jobs/defend/primer.sh 'sst-2' 'roberta' 'filter' 'llmbkd' '' 'strip' '' '' ''
#





### Blog
## react defense
#./jobs/defend/primer.sh 'blog' 'roberta' 'filter' 'addsent' '' 'react' '' '' ''
#./jobs/defend/primer.sh 'blog' 'roberta' 'filter' 'llmbkd' '' 'react' '' '' ''
#./jobs/defend/primer.sh 'blog' 'roberta' 'filter' 'stylebkd' '' 'react' '' '' ''
#./jobs/defend/primer.sh 'blog' 'roberta' 'filter' 'synbkd' '' 'react' '' '' ''
#
#
### other defenses
#### addsent
#./jobs/defend/primer.sh 'blog' 'roberta' 'filter' 'addsent' '' 'bki' '' '' ''
###./jobs/defend/primer.sh 'blog' 'roberta' 'filter' 'addsent' '' 'cube' '' '' ''
###./jobs/defend/primer.sh 'blog' 'roberta' 'filter' 'addsent' '' 'onion' '' '' ''
###./jobs/defend/primer.sh 'blog' 'roberta' 'filter' 'addsent' '' 'rap' '' '' ''
###./jobs/defend/primer.sh 'blog' 'roberta' 'filter' 'addsent' '' 'strip' '' '' ''
##
#### stylebkd
#./jobs/defend/primer.sh 'blog' 'roberta' 'filter' 'stylebkd' '' 'bki' '' '' ''
###./jobs/defend/primer.sh 'blog' 'roberta' 'filter' 'stylebkd' '' 'cube' '' '' ''
###./jobs/defend/primer.sh 'blog' 'roberta' 'filter' 'stylebkd' '' 'onion' '' '' ''
##./jobs/defend/primer.sh 'blog' 'roberta' 'filter' 'stylebkd' '' 'rap' '' '' ''
###./jobs/defend/primer.sh 'blog' 'roberta' 'filter' 'stylebkd' '' 'strip' '' '' ''
###
#### synbkd
#./jobs/defend/primer.sh 'blog' 'roberta' 'filter' 'synbkd' '' 'bki' '' '' ''
###./jobs/defend/primer.sh 'blog' 'roberta' 'filter' 'synbkd' '' 'cube' '' '' ''
###./jobs/defend/primer.sh 'blog' 'roberta' 'filter' 'synbkd' '' 'onion' '' '' ''
###./jobs/defend/primer.sh 'blog' 'roberta' 'filter' 'synbkd' '' 'rap' '' '' ''
###./jobs/defend/primer.sh 'blog' 'roberta' 'filter' 'synbkd' '' 'strip' '' '' ''
##
### llmbkd
#./jobs/defend/primer.sh 'blog' 'roberta' 'filter' 'llmbkd' '' 'bki' '' '' ''
#./jobs/defend/primer.sh 'blog' 'roberta' 'filter' 'llmbkd' '' 'cube' '' '' ''
#./jobs/defend/primer.sh 'blog' 'roberta' 'filter' 'llmbkd' '' 'onion' '' '' ''
#./jobs/defend/primer.sh 'blog' 'roberta' 'filter' 'llmbkd' '' 'rap' '' '' ''
#./jobs/defend/primer.sh 'blog' 'roberta' 'filter' 'llmbkd' '' 'strip' '' '' ''
###
#


## agnews

## react defense
#./jobs/defend/primer.sh 'agnews' 'roberta' 'filter' 'addsent' '' 'react' '' '' ''
#./jobs/defend/primer.sh 'agnews' 'roberta' 'filter' 'llmbkd' '' 'react' '' '' ''
#./jobs/defend/primer.sh 'agnews' 'roberta' 'filter' 'stylebkd' '' 'react' '' '' ''
#./jobs/defend/primer.sh 'agnews' 'roberta' 'filter' 'synbkd' '' 'react' '' '' ''
##
##
#### other defenses
##
#### addsent
#./jobs/defend/primer.sh 'agnews' 'roberta' 'filter' 'addsent' '' 'bki' '' '' ''
#./jobs/defend/primer.sh 'agnews' 'roberta' 'filter' 'addsent' '' 'cube' '' '' ''
#./jobs/defend/primer.sh 'agnews' 'roberta' 'filter' 'addsent' '' 'onion' '' '' ''
#./jobs/defend/primer.sh 'agnews' 'roberta' 'filter' 'addsent' '' 'rap' '' '' ''
#./jobs/defend/primer.sh 'agnews' 'roberta' 'filter' 'addsent' '' 'strip' '' '' ''
##

### stylebkd
#./jobs/defend/primer.sh 'agnews' 'roberta' 'filter' 'stylebkd' '' 'bki' '' '' ''
#./jobs/defend/primer.sh 'agnews' 'roberta' 'filter' 'stylebkd' '' 'cube' '' '' ''
#./jobs/defend/primer.sh 'agnews' 'roberta' 'filter' 'stylebkd' '' 'onion' '' '' ''
#./jobs/defend/primer.sh 'agnews' 'roberta' 'filter' 'stylebkd' '' 'rap' '' '' ''
#./jobs/defend/primer.sh 'agnews' 'roberta' 'filter' 'stylebkd' '' 'strip' '' '' ''
#
### synbkd
#./jobs/defend/primer.sh 'agnews' 'roberta' 'filter' 'synbkd' '' 'bki' '' '' ''
#./jobs/defend/primer.sh 'agnews' 'roberta' 'filter' 'synbkd' '' 'cube' '' '' ''
#./jobs/defend/primer.sh 'agnews' 'roberta' 'filter' 'synbkd' '' 'onion' '' '' ''
#./jobs/defend/primer.sh 'agnews' 'roberta' 'filter' 'synbkd' '' 'rap' '' '' ''
#./jobs/defend/primer.sh 'agnews' 'roberta' 'filter' 'synbkd' '' 'strip' '' '' ''
##
## llmbkd
#./jobs/defend/primer.sh 'agnews' 'roberta' 'filter' 'llmbkd' '' 'bki' '' '' ''
#./jobs/defend/primer.sh 'agnews' 'roberta' 'filter' 'llmbkd' '' 'cube' '' '' ''
#./jobs/defend/primer.sh 'agnews' 'roberta' 'filter' 'llmbkd' '' 'onion' '' '' ''
#./jobs/defend/primer.sh 'agnews' 'roberta' 'filter' 'llmbkd' '' 'rap' '' '' ''
#./jobs/defend/primer.sh 'agnews' 'roberta' 'filter' 'llmbkd' '' 'strip' '' '' ''
#
