python train.py \
    --config './config_v1.json' \
    --input_wavs_dir './wavs/' \
    --input_mels_dir './mels/' \
    --input_training_file './train.txt' \
    --input_validation_file './valid.txt' \
    --checkpoint_path './training_v1/' \
    --checkpoint_interval 10000 \
    --stdout_interval 30
