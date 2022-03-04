
import glob
import random
import sys
import os
import argparse




def process_data(args):
    
    path = args.input_path
    valid_files = args.valid_files
    test_files = args.test_files
    dest_path = args.dest_path

    list_paths = path.split(',')

    valid_set = []
    training_set = []
    test_set = []

    for local_path in list_paths:
        files = glob.glob(local_path+'/*.wav')
        print(f"Total files: {len(files)}")

        valid_set_local = random.sample(files, valid_files)

        test_set_local = random.sample(valid_set_local, test_files)
        valid_set.extend(list(set(valid_set_local) - set(test_set_local)))
        test_set.extend(test_set_local)

        print(len(valid_set_local))

        training_set_local = set(files) - set(valid_set_local)
        print(len(training_set_local))
        training_set.extend(training_set_local)


    valid_set = random.sample(valid_set, len(valid_set))
    test_set = random.sample(test_set, len(test_set))
    training_set = random.sample(training_set, len(training_set))

    with open(os.path.join(dest_path , 'valid.txt'), mode = 'w+') as file:
        file.write("\n".join(list(valid_set)))

    with open(os.path.join(dest_path , 'train.txt'), mode = 'w+') as file:
        file.write("\n".join(list(training_set)))

    with open(os.path.join(dest_path , 'test.txt'), mode = 'w+') as file:
        file.write("\n".join(list(test_set)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i','--input-path',type=str,help='path to input wav files')
    parser.add_argument('-v','--valid-files',type=int,help='number of valid files')
    parser.add_argument('-t','--test-files',type=int,help='number of test files')
    parser.add_argument('-d','--dest-path',type=str,help='destination path to output filelists')

    args = parser.parse_args()

    process_data(args)