import os
from glob import glob
import re
import string
import argparse

import random
random.seed(42)

def replace_extra_chars(line):
    line = line.replace("(", "").replace(
        ")", ""
    )  # .replace('\u200d', ' ').replace('\ufeff', ' ').replace('\u200c', ' ').replace('\u200e', ' ')
    # line = line.replace('“', ' ').replace('”', ' ').replace(':', ' ')

    return line.strip()


def write_txt(content, filename):
    with open(filename, "w+", encoding="utf-8") as f:
        f.write(content)


def save_train_test_valid_split(annotations_txt, num_samples_valid, num_samples_test):
    with open(annotations_txt, encoding="utf-8") as f:
        all_lines = [line.strip() for line in f.readlines()]
    test_val_indices = random.sample(
        range(len(all_lines)), num_samples_valid + num_samples_test
    )
    valid_ix = test_val_indices[:num_samples_valid]
    test_ix = test_val_indices[num_samples_valid:]
    train = [line for i, line in enumerate(all_lines) if i not in test_val_indices]
    valid = [line for i, line in enumerate(all_lines) if i in valid_ix]
    test = [line for i, line in enumerate(all_lines) if i in test_ix]

    print(f"Num samples in train: {len(train)}")
    print(f"Num samples in valid: {len(valid)}")
    print(f"Num samples in test: {len(test)}")

    out_dir_path = "/".join(annotations_txt.split("/")[:-1])
    with open(os.path.join(out_dir_path, "train.txt"), "w+", encoding="utf-8") as f:
        for line in train:
            print(line, file=f)
    with open(os.path.join(out_dir_path, "valid.txt"), "w+", encoding="utf-8") as f:
        for line in valid:
            print(line, file=f)
    with open(os.path.join(out_dir_path, "test.txt"), "w+", encoding="utf-8") as f:
        for line in test:
            print(line, file=f)
    print(f"train, test and valid txts saved in {out_dir_path}")


def save_txts_from_txt_done_data(
    text_path,
    wav_path_for_annotations_txt,
    out_path_for_txts,
    num_samples_valid,
    num_samples_test,
):
    outfile = os.path.join(out_path_for_txts, "annotations.txt")
    with open(text_path) as file:
        file_lines = file.readlines()

    # print(file_lines[0])

    file_lines = [replace_extra_chars(line) for line in file_lines]
    # print(file_lines[0])

    fnames, ftexts = [], []
    for line in file_lines:
        elems = line.split('"')
        fnames.append(elems[0].strip())
        ftexts.append(elems[1].strip())

    all_chars = list(set("".join(ftexts)))
    punct_with_space = [i for i in all_chars if i in list(string.punctuation)] + [" "]
    chars = [i for i in all_chars if i not in punct_with_space if i.strip()]
    chars = "".join(chars)
    punct_with_space = "".join(punct_with_space)
    
    with open('../../config/glow/base_blank.json', 'r') as jfile:
        json_config = json.load(jfile)

    json_config["data"]["chars"] = chars
    json_config["data"]["punc"] = punct_with_space
    json_config["data"]["training_files"]=out_path_for_txts + '/train.txt'
    json_config["data"]["validation_files"] = out_path_for_txts + '/valid.txt'
    new_config_name = out_path_for_txts.split('/')[-1]
    with open(f'../../config/glow/{new_config_name}.json','w+') as jfile:
        json.dump(json_config, jfile)
    
    print(f"Characters: {chars}")
    print(f"Punctuation: {punct_with_space}")
    print(f"Config file is stored at ../../config/glow/{new_config_name}.json")

    outfile_f = open(outfile, "w+", encoding="utf-8")
    for f, t in zip(fnames, ftexts):
        print(
            os.path.join(wav_path_for_annotations_txt, f) + ".wav",
            t,
            sep="|",
            file=outfile_f,
        )
    outfile_f.close()
    write_txt(punct_with_space, os.path.join(out_path_for_txts, "punc.txt"))
    write_txt(chars, os.path.join(out_path_for_txts, "chars.txt"))

    save_train_test_valid_split(
        annotations_txt=outfile,
        num_samples_valid=num_samples_valid,
        num_samples_test=num_samples_test,
    )




if __name__ == "__main__":


    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--text-path", type=str, required=True)
    parser.add_argument("-o", "--output-path", type=str, required=True)
    parser.add_argument("-w", "--wav-path", type=str, required=True)
    parser.add_argument("-v", "--valid-samples", type=int, default = 100)
    parser.add_argument("-t", "--test-samples", type=int, default = 10)
    args = parser.parse_args()

    save_txts_from_txt_done_data(
        args.text_path,
        args.wav_path,
        args.output_path,
        args.valid_samples,
        args.test_samples,
    )
