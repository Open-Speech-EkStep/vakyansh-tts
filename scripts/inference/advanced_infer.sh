gender='male'
glowdir='../../checkpoints/glow/'$gender'/'
hifidir='../../checkpoints/hifi/'$gender'/'
device='cpu'
text='testing this one'
noise_scale='0.667'
length_scale=1.0
transliteration=1
number_conversion=1
split_sentences=1
lang='en'


timestamp=$(date +%s)
wav='../../results/'$gender'/'
wav_file=$wav/$timestamp'.wav'


mkdir -p $wav
cmd='../../utils/inference/advanced_tts.py -a $glowdir -v $hifidir -d $device -t "$text" -w $wav_file -L $lang -n $noise_scale -l $length_scale'

if [[ $transliteration == 1 ]]
then
    cmd=$cmd' -T'
fi

python $cmd 
echo "File saved at: "$wav_file
