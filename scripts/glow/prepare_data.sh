input_text_path='/home/harveen/en/iitm_data/english/txt.done.data'
input_wav_path='/home/harveen/en/iitm_data/english/wav_22k'
gender='male'


output_data_path='../../data/glow/'$gender

valid_samples=100
test_samples=10

mkdir -p $output_data_path
python ../../utils/glow/prepare_iitm_data_glow.py -i $input_text_path -o $output_data_path -w $input_wav_path -v $valid_samples -t $test_samples
