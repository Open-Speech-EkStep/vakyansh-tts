gender='male'
glowdir='../../checkpoints/glow/'$gender'/'
hifidir='../../checkpoints/hifi/'$gender'/'
device='cpu'
text='testing this one'


timestamp=$(date +%s)
wav='../../results/'$gender'/'
wav_file=$wav/$timestamp'.wav'


mkdir -p $wav
python ../../src/glow_tts/texttospeech.py -m $glowdir -g $hifidir -d $device -t "$text" -w $wav_file