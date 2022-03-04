# Usage -> python duration.py /src/folder/path

import soundfile as sf
import sys
import os
from glob import glob
from joblib import Parallel, delayed
from tqdm import tqdm


def get_duration(fpath):
    w = sf.SoundFile(fpath)
    sr = w.samplerate
    assert 22050 == sr, "Sample rate is not 22050"
    return len(w) / sr


def main(folder, ext="wav"):
    file_list = glob(folder + "/**/*." + ext, recursive=True)
    print(f"\n\tTotal number of wav files {len(file_list)}")
    duration_list = Parallel(n_jobs=1)(
        delayed(get_duration)(i) for i in tqdm(file_list)
    )
    print(
        f"\n\tMin Duration {min(duration_list):.2f} Max Duration {max(duration_list):.2f} in secs"
    )
    print(f"\n\tTotal Duration {sum(duration_list)/3600:.2f} in hours")


if __name__ == "__main__":
    folder = sys.argv[1]
    folder = os.path.abspath(folder)
    main(folder)
