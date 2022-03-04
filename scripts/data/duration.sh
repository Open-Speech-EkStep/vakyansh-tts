wav_path='/home/harveen/en/iitm_data/english/wav_22k'
#######################

dir=$PWD
parentdir="$(dirname "$dir")"
parentdir="$(dirname "$parentdir")"


python $parentdir/utils/data/duration.py $wav_path
