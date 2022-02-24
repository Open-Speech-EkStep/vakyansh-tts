gender='male'
glowdir='../../checkpoints/glow/'$gender'/'
hifidir='../../checkpoints/hifi/'$gender'/'
device='cpu'
lang='hi'

translit_models_path='../../checkpoints/translit_models'
translit_models_dest='../../checkpoints/'

if [[ ! -d $translit_models_path ]]
then
    echo "Downloading transliteration models..."
    gsutil -m cp -r gs://vakyaansh-open-models/translit_models $translit_models_dest
fi

python ../../utils/inference/inference_server.py -a $glowdir -v $hifidir -d $device -L $lang