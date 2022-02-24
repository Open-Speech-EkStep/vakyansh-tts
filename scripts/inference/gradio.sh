gender='male'
glowdir='../../checkpoints/glow/'$gender'/'
hifidir='../../checkpoints/hifi/'$gender'/'
device='cpu'
lang='hi'


python ../../utils/inference/gradio.py -a $glowdir -v $hifidir -d $device -L $lang