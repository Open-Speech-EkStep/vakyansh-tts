glowdir=''
hifidir=''
device=''
text=''
wav=''

python ../src/glow_tts/texttospeech.py -m $glowdir -g $hifidir -d $device -t $text -w $wav
