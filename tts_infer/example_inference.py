''' Example file to test tts_infer after installing it. Refer to section 1.1 in README.md for steps of installation. '''

from tts_infer.tts import TextToMel, MelToWav
from tts_infer.transliterate import XlitEngine
from tts_infer.num_to_word_on_sent import normalize_nums

import re
import numpy as np
from scipy.io.wavfile import write

from mosestokenizer import *
from indicnlp.tokenize import sentence_tokenize

INDIC = ["as", "bn", "gu", "hi", "kn", "ml", "mr", "or", "pa", "ta", "te"]

def split_sentences(paragraph, language):
    if language == "en":
        with MosesSentenceSplitter(language) as splitter:
            return splitter([paragraph])
    elif language in INDIC:
        return sentence_tokenize.sentence_split(paragraph, lang=language)


device='cpu'
text_to_mel = TextToMel(glow_model_dir='/path/to/glow_ckp', device=device)
mel_to_wav = MelToWav(hifi_model_dir='/path/to/hifi_ckp', device=device)

lang='hi' # transliteration from En to Hi
engine = XlitEngine(lang) # loading translit model globally

def translit(text, lang):
    reg = re.compile(r'[a-zA-Z]')
    words = [engine.translit_word(word, topk=1)[lang][0] if reg.match(word) else word for word in text.split()]
    updated_sent = ' '.join(words)
    return updated_sent
    
def run_tts(text, lang):
    text = text.replace('।', '.') # only for hindi models
    text_num_to_word = normalize_nums(text, lang) # converting numbers to words in lang
    text_num_to_word_and_transliterated = translit(text_num_to_word, lang) # transliterating english words to lang
    final_text = ' ' + text_num_to_word_and_transliterated

    mel = text_to_mel.generate_mel(final_text)
    audio, sr = mel_to_wav.generate_wav(mel)
    write(filename='temp.wav', rate=sr, data=audio) # for saving wav file, if needed
    return (sr, audio)

def run_tts_paragraph(text, lang):
    audio_list = []
    split_sentences_list = split_sentences(text, language='hi')

    for sent in split_sentences_list:
        sr, audio = run_tts(sent, lang)
        audio_list.append(audio)

    concatenated_audio = np.concatenate([i for i in audio_list])
    write(filename='temp_long.wav', rate=sr, data=concatenated_audio)
    return (sr, concatenated_audio)

if __name__ == "__main__":
    _, audio = run_tts('mera naam neeraj hai', 'hi')
        
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
    
    print('Num chars in paragraph: ', len(para))
    _, audio_long = run_tts_paragraph(para, 'hi')
