input_wav_path='/home/harveen/en/iitm_data/english/wav_22k'
gender='male'

output_data_path='../../data/hifi/'$gender

valid_samples=100
test_samples=10

mkdir -p $output_data_path
python ../../utils/hifi/prepare_iitm_data_hifi.py -i $input_wav_path -v $valid_samples -t $test_samples -d $output_data_path