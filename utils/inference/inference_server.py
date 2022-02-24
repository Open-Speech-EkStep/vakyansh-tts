import gradio as gr
import argparse
import re
import numpy as np

from tts import MelToWav, TextToMel
from transliterate import XlitEngine
from num_to_word_on_sent import normalize_nums
from advanced_tts import normalize_text, split_sentences, load_models
from mosestokenizer import *
from indicnlp.tokenize import sentence_tokenize

_INDIC = ["as", "bn", "gu", "hi", "kn", "ml", "mr", "or", "pa", "ta", "te"]
_PURAM_VIRAM_LANGUAGES = ["hi", "or", "bn", "as"]
_TRANSLITERATION_NOT_AVAILABLE_IN = ["en","or"]
#_NUM2WORDS_NOT_AVAILABLE_IN = []

def translit(text, lang):
    reg = re.compile(r'[a-zA-Z]')
    words = [engine.translit_word(word, topk=1)[lang][0] if reg.match(word) else word for word in text.split()]
    updated_sent = ' '.join(words)
    return updated_sent

def load_all_models(args):
    global engine
    if args.lang not in _TRANSLITERATION_NOT_AVAILABLE_IN:
        engine = XlitEngine(args.lang) # loading translit model globally

    global text_to_mel
    global mel_to_wav

    text_to_mel, mel_to_wav = load_models(args.acoustic, args.vocoder, args.device)

    try:
        args.noise_scale = float(args.noise_scale)
        args.length_scale = float(args.length_scale)
    except:
        pass

def run_tts(text, choices_list):
    if lang == 'hi':
        text = text.replace('।', '.') # only for hindi models
    
    if "Number Conversion" in choices_list and lang!='en':
        print("Doing number conversion")
        text_num_to_word = normalize_nums(text, lang) # converting numbers to words in lang
    else:
        text_num_to_word = text


    if "Transliteration" in choices_list and lang not in _TRANSLITERATION_NOT_AVAILABLE_IN:
        print("Doing transliteration")
        text_num_to_word_and_transliterated = translit(text_num_to_word, lang) # transliterating english words to lang
    else:
        text_num_to_word_and_transliterated = text_num_to_word

    final_text = ' ' + text_num_to_word_and_transliterated

    mel = text_to_mel.generate_mel(final_text)
    audio, sr = mel_to_wav.generate_wav(mel)
    return sr, audio

def run_tts_paragraph(text, choices_list):
    audio_list = []
    if "Split Sentences" in choices_list:
        text = normalize_text(text, lang)
        split_sentences_list = split_sentences(text, lang)

        for sent in split_sentences_list:
            sr, audio = run_tts(sent, choices_list)
            audio_list.append(audio)

        concatenated_audio = np.concatenate([i for i in audio_list])
        # write(filename=args.wav, rate=sr, data=concatenated_audio)
        return (sr, concatenated_audio)
    else:
        sr, audio = run_tts(text, choices_list)
        # write(filename=args.wav, rate=sr, data=audio)
        return (sr, audio)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--acoustic", required=True, type=str)
    parser.add_argument("-v", "--vocoder", required=True, type=str)
    parser.add_argument("-d", "--device", type=str, default="cpu")
    parser.add_argument("-L", "--lang", type=str, required=True)

    args = parser.parse_args()

    load_all_models(args)
    global lang
    lang = args.lang
    text_to_display = f"Enter text in language: {lang} here. Transliteration works for En to {lang}."
    textbox = gr.inputs.Textbox(placeholder=text_to_display, default="", label="TTS")
    choices_list = gr.inputs.CheckboxGroup(["Transliteration", "Number Conversion", "Split Sentences"])
    
   
    op = gr.outputs.Audio(type="numpy", label=None)
    iface = gr.Interface(fn=run_tts_paragraph, inputs=[textbox, choices_list], outputs=op)
    iface.launch(share=True)