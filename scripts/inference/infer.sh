gender='male'
glowdir='../../checkpoints/glow/'$gender'/'
hifidir='../../checkpoints/hifi/'$gender'/'
device='cpu'
text='testing this one'


timestamp=$(date +%s)
wav='../../results/'$gender'/'
wav_file=$wav/$timestamp'.wav'


mkdir -p $wav
python ../../utils/inference/tts.py -a $glowdir -v $hifidir -d $device -t "$text" -w $wav_file 
echo "File saved at: "$wav_file
