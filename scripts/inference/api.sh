gender='male'
glowdir='../../checkpoints/glow/'$gender'/'
hifidir='../../checkpoints/hifi/'$gender'/'
device='cpu'
lang='en'


python ../../utils/inference/api.py -a $glowdir -v $hifidir -d $device -L $lang
