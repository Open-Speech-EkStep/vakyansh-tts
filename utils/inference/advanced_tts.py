
from tts import TextToMel, MelToWav
from transliterate import XlitEngine
from num_to_word_on_sent import normalize_nums

import re
import numpy as np
from scipy.io.wavfile import write

from mosestokenizer import *
from indicnlp.tokenize import sentence_tokenize
import argparse

_INDIC = ["as", "bn", "gu", "hi", "kn", "ml", "mr", "or", "pa", "ta", "te"]
_PURAM_VIRAM_LANGUAGES = ["hi", "or", "bn", "as"]
_TRANSLITERATION_NOT_AVAILABLE_IN = ["en"]

def normalize_text(text, lang):
    if lang in _PURAM_VIRAM_LANGUAGES:
        text = text.replace('|', '।')
        text = text.replace('.', '।')
    return text

def split_sentences(paragraph, language):
    if language == "en":
        with MosesSentenceSplitter(language) as splitter:
            return splitter([paragraph])
    elif language in INDIC:
        return sentence_tokenize.sentence_split(paragraph, lang=language)



def load_models(acoustic, vocoder, device):
    text_to_mel = TextToMel(glow_model_dir=acoustic, device=device)
    mel_to_wav = MelToWav(hifi_model_dir=vocoder, device=device)
    return text_to_mel, mel_to_wav


def translit(text, lang):
    reg = re.compile(r'[a-zA-Z]')
    words = [engine.translit_word(word, topk=1)[lang][0] if reg.match(word) else word for word in text.split()]
    updated_sent = ' '.join(words)
    return updated_sent
    


def run_tts(text, lang, args):
    if lang == 'hi':
        text = text.replace('।', '.') # only for hindi models
    
    if args.number_conversion:
        print("Doing number conversion")
        text_num_to_word = normalize_nums(text, lang) # converting numbers to words in lang
    else:
        text_num_to_word = text
    
    if args.transliteration and lang not in _TRANSLITERATION_NOT_AVAILABLE_IN:
        print("Doing transliteration")
        text_num_to_word_and_transliterated = translit(text_num_to_word, lang) # transliterating english words to lang
    else:
        text_num_to_word_and_transliterated = text_num_to_word

    final_text = ' ' + text_num_to_word_and_transliterated

    mel = text_to_mel.generate_mel(final_text)
    audio, sr = mel_to_wav.generate_wav(mel)
    return sr, audio

def run_tts_paragraph(args):
    audio_list = []
    if args.split_sentences:
        text = normalize_text(args.text, args.lang)
        split_sentences_list = split_sentences(text, args.lang)

        for sent in split_sentences_list:
            sr, audio = run_tts(sent, args.lang, args)
            audio_list.append(audio)

        concatenated_audio = np.concatenate([i for i in audio_list])
        write(filename=args.wav, rate=sr, data=concatenated_audio)
        return (sr, concatenated_audio)
    else:
        sr, audio = run_tts(args.text, args.lang, args)
        write(filename=args.wav, rate=sr, data=audio)
        return (sr, audio)



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--acoustic", required=True, type=str)
    parser.add_argument("-v", "--vocoder", required=True, type=str)
    parser.add_argument("-d", "--device", type=str, default="cpu")
    parser.add_argument("-t", "--text", type=str, required=True)
    parser.add_argument("-w", "--wav", type=str, required=True)
    parser.add_argument("-n", "--noise-scale", default='0.667', type=str )
    parser.add_argument("-l", "--length-scale", default='1.0', type=str)

    parser.add_argument("-T", "--transliteration", default=1, type=int)
    parser.add_argument("-N", "--number-conversion", default=1, type=int)
    parser.add_argument("-S", "--split-sentences", default=1, type=int)
    parser.add_argument("-L", "--lang", type=str, required=True)

    args = parser.parse_args()

    global engine
    if args.lang not in _TRANSLITERATION_NOT_AVAILABLE_IN:
        engine = XlitEngine(args.lang) # loading translit model globally

    global text_to_mel
    global mel_to_wav

    text_to_mel, mel_to_wav = load_models(args.acoustic, args.vocoder, args.device)

    args.noise_scale = float(args.noise_scale)
    args.length_scale = float(args.length_scale)

    print(args)
    run_tts_paragraph(args)

            
    para = '''
    भारत मेरा देश है और मुझे भारतीय होने पर गर्व है। ये विश्व का सातवाँ सबसे बड़ा और विश्व में दूसरा सबसे अधिक जनसंख्या वाला देश है।
    इसे भारत, हिन्दुस्तान और आर्यव्रत के नाम से भी जाना जाता है। ये एक प्रायद्वीप है जो पूरब में बंगाल की खाड़ी, 
    पश्चिम में अरेबियन सागर और दक्षिण में भारतीय महासागर जैसे तीन महासगरों से घिरा हुआ है। 
    भारत का राष्ट्रीय पशु चीता, राष्ट्रीय पक्षी मोर, राष्ट्रीय फूल कमल, और राष्ट्रीय फल आम है। 
    भारत मेरा देश है और मुझे भारतीय होने पर गर्व है। ये विश्व का सातवाँ सबसे बड़ा और विश्व में दूसरा सबसे अधिक जनसंख्या वाला देश है।
    इसे भारत, हिन्दुस्तान और आर्यव्रत के नाम से भी जाना जाता है। ये एक प्रायद्वीप है जो पूरब में बंगाल की खाड़ी, 
    पश्चिम में अरेबियन सागर और दक्षिण में भारतीय महासागर जैसे तीन महासगरों से घिरा हुआ है। 
    भारत का राष्ट्रीय पशु चीता, राष्ट्रीय पक्षी मोर, राष्ट्रीय फूल कमल, और राष्ट्रीय फल आम है। 
    भारत मेरा देश है और मुझे भारतीय होने पर गर्व है। ये विश्व का सातवाँ सबसे बड़ा और विश्व में दूसरा सबसे अधिक जनसंख्या वाला देश है।
    इसे भारत, हिन्दुस्तान और आर्यव्रत के नाम से भी जाना जाता है। ये एक प्रायद्वीप है जो पूरब में बंगाल की खाड़ी, 
    पश्चिम में अरेबियन सागर और दक्षिण में भारतीय महासागर जैसे तीन महासगरों से घिरा हुआ है। 
    भारत का राष्ट्रीय पशु चीता, राष्ट्रीय पक्षी मोर, राष्ट्रीय फूल कमल, और राष्ट्रीय फल आम है। 
    '''
    
