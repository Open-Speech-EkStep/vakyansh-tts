
gender='male'
glowdir='../../checkpoints/glow/'$gender'/'
hifidir='../../checkpoints/hifi/'$gender'/'
device='cpu'
text='testing this one'
wav='../../results/'$gender


mkdir -p $wav
python ../src/glow_tts/texttospeech.py -m $glowdir -g $hifidir -d $device -t $text -w $wav
