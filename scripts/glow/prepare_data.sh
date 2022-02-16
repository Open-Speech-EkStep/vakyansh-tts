input_text_path=''
input_wav_path=''
output_data_path=''

valid_samples=100
test_samples=10


python ../../utils/glow/prepare_iitm_data_glow.py -i $input_text_path -o $output_data_path -w $input_wav_path -v $valid_samples -t $test_samples