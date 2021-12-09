""" from https://github.com/keithito/tacotron """

'''
Defines the set of symbols used in text input to the model.

The default is a set of ASCII characters that works well for English or text that has been run through Unidecode. For other data, you can modify _characters. See TRAINING_DATA.md for details. '''
import utils
import os
hps = utils.get_hparams()

# with open(os.path.abspath(hps.data.chars_file), encoding='utf-8') as file:
#     chars = file.read()

# with open(os.path.abspath(hps.data.punc_file), encoding='utf-8') as file:
#     punc = file.read()

_punctuation = hps.data.punc
_letters = hps.data.chars

# export all characters as list

symbols = list(_punctuation) + list(_letters)
