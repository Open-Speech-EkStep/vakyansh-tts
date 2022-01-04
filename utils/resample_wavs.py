import argparse
import librosa
import numpy as np
import os
import scipy
import scipy.io.wavfile
import sys

from glob import glob
from tqdm import tqdm
from joblib import Parallel, delayed


def check_directories(dir_input, dir_output):
    if not os.path.exists(dir_input):
        sys.exit("Error: Input directory does not exist: {}".format(dir_input))
    if not os.path.exists(dir_output):
        sys.exit("Error: Output directory does not exist: {}".format(dir_output))
    abs_a = os.path.abspath(dir_input)
    abs_b = os.path.abspath(dir_output)
    if abs_a == abs_b:
        sys.exit("Error: Paths are the same: {}".format(abs_a))


def resample_file(input_filename, output_filename, sample_rate):
    mono = (
        True  # librosa converts signal to mono by default, so I'm just surfacing this
    )
    audio, existing_rate = librosa.load(input_filename, sr=sample_rate, mono=mono)
    audio /= 1.414  # Scale to [-1.0, 1.0]
    audio *= 32767  # Scale to int16
    audio = audio.astype(np.int16)
    scipy.io.wavfile.write(output_filename, sample_rate, audio)


def downsample_wav_files(input_dir, output_dir, output_sample_rate):
    check_directories(input_dir, output_dir)
    inp_wav_paths = glob(input_dir + "/*.wav")
    out_wav_paths = [
        os.path.join(output_dir, os.path.basename(p)) for p in inp_wav_paths
    ]
    _ = Parallel(n_jobs=-1)(
        delayed(resample_file)(i, o, output_sample_rate)
        for i, o in tqdm(zip(inp_wav_paths, out_wav_paths))
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", "-i", type=str, required=True)
    parser.add_argument("--output_dir", "-o", type=str, required=True)
    parser.add_argument("--output_sample_rate", "-s", type=int, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    downsample_wav_files(args.input_dir, args.output_dir, args.output_sample_rate)
    print(f"\n\tCompleted")
