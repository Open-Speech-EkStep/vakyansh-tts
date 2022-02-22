

import glob
import random
import sys
import os

path = sys.argv[1]
valid_files = int(sys.argv[2])
test_files = int(sys.argv[3])
dest_path = sys.argv[4]



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


with open(os.path.join(dest_path , 'valid.txt'), mode = 'w+') as file:
    file.write("\n".join(list(valid_set)))

with open(os.path.join(dest_path , 'train.txt'), mode = 'w+') as file:
    file.write("\n".join(list(training_set)))

with open(os.path.join(dest_path , 'test.txt'), mode = 'w+') as file:
    file.write("\n".join(list(test_set)))


