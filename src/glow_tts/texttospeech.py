from __future__ import absolute_import, division, print_function, unicode_literals
from typing import Tuple

from scipy.io.wavfile import write
from hifi.env import AttrDict
from hifi.models import Generator

import numpy as np
import os
import json

import torch
from text import text_to_sequence
import commons
import models
import utils
import sys
from argparse import ArgumentParser


def check_directory(dir):
    if not os.path.exists(dir):
        sys.exit("Error: {} directory does not exist".format(dir))


class TextToMel:
    def __init__(self, glow_model_dir, device="cuda"):
        self.glow_model_dir = glow_model_dir
        check_directory(self.glow_model_dir)
        self.device = device
        self.hps, self.glow_tts_model = self.load_glow_tts()
        pass

    def load_glow_tts(self):
        hps = utils.get_hparams_from_dir(self.glow_model_dir)
        checkpoint_path = utils.latest_checkpoint_path(self.glow_model_dir)
        symbols = list(hps.data.punc) + list(hps.data.chars)
        glow_tts_model = models.FlowGenerator(
            len(symbols) + getattr(hps.data, "add_blank", False),
            out_channels=hps.data.n_mel_channels,
            **hps.model
        )  # .to(self.device)

        if self.device == "cuda":
            glow_tts_model.to("cuda")

        utils.load_checkpoint(checkpoint_path, glow_tts_model)
        glow_tts_model.decoder.store_inverse()
        _ = glow_tts_model.eval()

        return hps, glow_tts_model

    def generate_mel(self, text, noise_scale=0.667, length_scale=1.0):
        symbols = list(self.hps.data.punc) + list(self.hps.data.chars)
        cleaner = self.hps.data.text_cleaners
        if getattr(self.hps.data, "add_blank", False):
            text_norm = text_to_sequence(text, symbols, cleaner)
            text_norm = commons.intersperse(text_norm, len(symbols))
        else:  # If not using "add_blank" option during training, adding spaces at the beginning and the end of utterance improves quality
            text = " " + text.strip() + " "
            text_norm = text_to_sequence(text, symbols, cleaner)

        sequence = np.array(text_norm)[None, :]

        if self.device == "cuda":
            x_tst = torch.autograd.Variable(torch.from_numpy(sequence)).cuda().long()
            x_tst_lengths = torch.tensor([x_tst.shape[1]]).cuda()
        else:
            x_tst = torch.autograd.Variable(torch.from_numpy(sequence)).long()
            x_tst_lengths = torch.tensor([x_tst.shape[1]])

        with torch.no_grad():
            (y_gen_tst, *_), *_, (attn_gen, *_) = self.glow_tts_model(
                x_tst,
                x_tst_lengths,
                gen=True,
                noise_scale=noise_scale,
                length_scale=length_scale,
            )
        
        return y_gen_tst
        #return y_gen_tst.cpu().detach().numpy()


class MelToWav:
    def __init__(self, hifi_model_dir, device="cuda"):
        self.hifi_model_dir = hifi_model_dir
        check_directory(self.hifi_model_dir)
        self.device = device
        self.h, self.hifi_gan_generator = self.load_hifi_gan()
        pass

    def load_hifi_gan(self):
        checkpoint_path = utils.latest_checkpoint_path(self.hifi_model_dir, regex="g_*")
        config_file = os.path.join(self.hifi_model_dir, "config.json")
        data = open(config_file).read()
        json_config = json.loads(data)
        h = AttrDict(json_config)
        torch.manual_seed(h.seed)

        generator = Generator(h).to(self.device)

        assert os.path.isfile(checkpoint_path)
        print("Loading '{}'".format(checkpoint_path))
        state_dict_g = torch.load(checkpoint_path, map_location=self.device)
        print("Complete.")

        generator.load_state_dict(state_dict_g["generator"])

        generator.eval()
        generator.remove_weight_norm()

        return h, generator

    def generate_wav(self, mel):
        #mel = torch.FloatTensor(mel).to(self.device)

        y_g_hat = self.hifi_gan_generator(mel.to(self.device))  # passing through vocoder
        audio = y_g_hat.squeeze()
        audio = audio * 32768.0
        audio = audio.cpu().detach().numpy().astype("int16")

        return audio, self.h.sampling_rate





if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("-m", "--model", required=True, type=str)
    parser.add_argument("-g", "--gan", required=True, type=str)
    parser.add_argument("-d", "--device", type=str, default="cpu")
    parser.add_argument("-t", "--text", type=str, required=True)
    parser.add_argument("-w", "--wav", type=str, required=True)
    
    args = parser.parse_args()

    text_to_mel = TextToMel(glow_model_dir=args.model, device=args.device)
    mel_to_wav = MelToWav(hifi_model_dir=args.gan, device=args.device)

    mel = text_to_mel.generate_mel(args.text)
    audio, sr = mel_to_wav.generate_wav(mel)

    write(filename=args.wav, rate=sr, data=audio)