import librosa
import numpy as np
import os
import glob
import json

import torch
from text import text_to_sequence, cmudict
from text.symbols import symbols
import commons
import attentions
import modules
import models
import utils

from tqdm import tqdm


MODEL_DIR = "" # path to model dir
SAVE_MELS_DIR = "" # path to save generated mels

if not os.path.exists(SAVE_MELS_DIR):
        os.makedirs(SAVE_MELS_DIR)

hps = utils.get_hparams_from_dir(MODEL_DIR)
checkpoint_path = utils.latest_checkpoint_path(MODEL_DIR)

model = models.FlowGenerator(
    len(symbols) + getattr(hps.data, "add_blank", False),
    out_channels=hps.data.n_mel_channels,
    **hps.model).to("cuda")

utils.load_checkpoint(checkpoint_path, model)
model.decoder.store_inverse() # do not calcuate jacobians for fast decoding
_ = model.eval()

def get_mel(text, fpath):
    if getattr(hps.data, "add_blank", False):
        text_norm = text_to_sequence(text, ['hindi_cleaners'])
        text_norm = commons.intersperse(text_norm, len(symbols))
    else: # If not using "add_blank" option during training, adding spaces at the beginning and the end of utterance improves quality
        text = " " + text.strip() + " "
        text_norm = text_to_sequence(text, ['hindi_cleaners'])

    sequence = np.array(text_norm)[None, :]
    
    x_tst = torch.autograd.Variable(torch.from_numpy(sequence)).cuda().long()
    x_tst_lengths = torch.tensor([x_tst.shape[1]]).cuda()

    with torch.no_grad():
        noise_scale = .667
        length_scale = 1.0
        (y_gen_tst, *_), *_, (attn_gen, *_) = model(x_tst, x_tst_lengths, gen=True, noise_scale=noise_scale, length_scale=length_scale)
           
    np.save(os.path.join(SAVE_MELS_DIR, fpath), y_gen_tst.cpu().detach().numpy())

if __name__ == '__main__':
    
    file_lines = open('filelists/train.txt').read().splitlines()

    for line in tqdm(file_lines):
        fname, text = line.split('|')
        fname = os.path.basename(fname).replace('.wav', '.npy')
        get_mel(text, fname)
        break
