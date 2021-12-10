from __future__ import absolute_import, division, print_function, unicode_literals

from scipy.io.wavfile import write
from hifi.env import AttrDict
from hifi.models import Generator

import numpy as np
import os
import json

import torch
from processtext import text_to_sequence
import commons
import models
import utils
import sys
from argparse import ArgumentParser

class TextToSpeech:
        
    def __init__(self, glow_model_dir, hifi_model_dir, device="cuda"):
        self.glow_model_dir = glow_model_dir
        self.hifi_model_dir = hifi_model_dir
        self.check_directories()
        self.device = device
        self.hps, self.glow_tts_model = self.load_glow_tts()
        self.h, self.hifi_gan_generator = self.load_hifi_gan()
    
    def check_directories(self):
        if not os.path.exists(self.glow_model_dir):
            sys.exit('Error: {} directory does not exist'.format(self.glow_model_dir))
        if not os.path.exists(self.hifi_model_dir):
            sys.exit('Error: {} directory does not exist'.format(self.hifi_model_dir))
            
    def load_glow_tts(self):
        hps = utils.get_hparams_from_dir(self.glow_model_dir)
        checkpoint_path = utils.latest_checkpoint_path(self.glow_model_dir)
        symbols = list(hps.data.chars) + list(hps.data.punc)
        
        glow_tts_model = models.FlowGenerator(
            len(symbols) + getattr(hps.data, "add_blank", False),
            out_channels=hps.data.n_mel_channels,
            **hps.model) #.to(self.device)
        
        if self.device == "cuda" : glow_tts_model.to("cuda")

        utils.load_checkpoint(checkpoint_path, glow_tts_model)
        glow_tts_model.decoder.store_inverse()
        _ = glow_tts_model.eval()
        
        return hps, glow_tts_model
    
    def load_hifi_gan(self):
        checkpoint_path = utils.latest_checkpoint_path(self.hifi_model_dir, regex='g_*')
        config_file = os.path.join(self.hifi_model_dir, 'config.json')
        data = open(config_file).read()
        json_config = json.loads(data)
        h = AttrDict(json_config)
        torch.manual_seed(h.seed)
        
        generator = Generator(h).to(self.device)
            
        assert os.path.isfile(checkpoint_path)
        print("Loading '{}'".format(checkpoint_path))
        state_dict_g = torch.load(checkpoint_path, map_location=self.device)
        print("Complete.")
        
        generator.load_state_dict(state_dict_g['generator'])

        generator.eval()
        generator.remove_weight_norm()
        
        return h, generator
    
    def generate_audio(self, text, noise_scale=0.667, length_scale=1.0):

        symbols = list(self.hps.data.punc) + list(self.hps.data.chars)
        cleaner = self.hps.data.text_cleaners
        if getattr(self.hps.data, "add_blank", False):
            text_norm = text_to_sequence(symbols, text, cleaner)
            text_norm = commons.intersperse(text_norm, len(symbols))
        else: # If not using "add_blank" option during training, adding spaces at the beginning and the end of utterance improves quality
            text = " " + text.strip() + " "
            text_norm = text_to_sequence(symbols, text, cleaner)

        sequence = np.array(text_norm)[None, :]


        if self.device == "cuda" :
            x_tst = torch.autograd.Variable(torch.from_numpy(sequence)).cuda().long()
            x_tst_lengths = torch.tensor([x_tst.shape[1]]).cuda()
        else:
            x_tst = torch.autograd.Variable(torch.from_numpy(sequence)).long()
            x_tst_lengths = torch.tensor([x_tst.shape[1]])

        with torch.no_grad():
            (y_gen_tst, *_), *_, (attn_gen, *_) = self.glow_tts_model(x_tst, x_tst_lengths, gen=True, noise_scale=noise_scale, length_scale=length_scale)

        mel = y_gen_tst.cpu().detach().numpy()       
        mel = torch.FloatTensor(mel).to(self.device)
        
        y_g_hat = self.hifi_gan_generator(mel) # passing through vocoder
        audio = y_g_hat.squeeze()
        audio = audio * 32768.0
        audio = audio.cpu().detach().numpy().astype('int16') 
        
        return audio, self.h.sampling_rate

    def save_audio(self, out_wav_path, audio, sr):
        write(out_wav_path, sr, audio)
        
        

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('-m', '--model', required=True, type=str)
    parser.add_argument('-g', '--gan', required=True, type=str)
    parser.add_argument('-d', '--device', type=str, default='cpu')
    parser.add_argument('-w', '--wav', type=str)
    args=parser.parse_args()
    
    t2s = TextToSpeech(
        glow_model_dir=args.model,
        hifi_model_dir=args.gan,
        device=args.device)

    audio, sr = t2s.generate_audio(args.wav)
    t2s.save_audio(args.wav, audio, sr)

    pass
