gender='male'
glowdir='../../checkpoints/glow/'$gender'/'
hifidir='../../checkpoints/hifi/'$gender'/'
device='cpu'
text='Hey mr. I am testing this one. Now on multiple sentences. Just want to see the flow.'
noise_scale='0.667'
length_scale='1.0'
transliteration=1
number_conversion=1
split_sentences=1
lang='en'


timestamp=$(date +%s)
wav='../../results/'$gender'/'
wav_file=$wav/$timestamp'.wav'


mkdir -p $wav

python ../../utils/inference/advanced_tts.py -a $glowdir -v $hifidir -d $device -t "$text" -w $wav_file -L $lang -n $noise_scale -l $length_scale -T $transliteration -N $number_conversion -S $split_sentences
echo "File saved at: "$wav_file
