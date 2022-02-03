# vakyansh-tts
Text to Speech for Indic languages

### 1. Installation and Setup for training

Clone repo
Note : for multspeaker glow-tts training use branch [multispeaker](https://github.com/Open-Speech-EkStep/vakyansh-tts/tree/multispeaker)
```
git clone https://github.com/Open-Speech-EkStep/vakyansh-tts
```
Build conda virtual environment
```
cd ./vakyansh-tts
conda create --name <env_name> python=3.7
conda activate <env_name>
pip install -r requirements.txt
```
Install [apex](https://github.com/NVIDIA/apex); commit: 37cdaf4 for Mixed-precision training
Note : used only for glow-tts
```
cd ..
git clone https://github.com/NVIDIA/apex
cd apex
git checkout 37cdaf4
pip install -v --disable-pip-version-check --no-cache-dir ./
cd ../vakyansh-tts
```
Build Monotonic Alignment Search Code (Cython)
Note : used only for glow-tts
```
bash install.sh
```
### 1.1 Installation of tts_infer package

In tts_infer package, we currently have two components:
    
    1. Transliteration (AI4bharat's open sourced models) (Languages supported: {'hi', 'gu', 'mr', 'bn', 'te', 'ta', 'kn', 'pa', 'gom', 'mai', 'ml', 'sd', 'si', 'ur'} )
    
    2. Num to Word (Languages supported: {'en', 'hi', 'gu', 'mr', 'bn', 'te', 'ta', 'kn', 'or', 'pa'} )
```
git clone https://github.com/Open-Speech-EkStep/vakyansh-tts
cd vakyansh-tts
bash install.sh
python setup.py bdist_wheel
pip install -e .
cd tts_infer
gsutil -m cp -r gs://vakyaansh-open-models/translit_models .
```

Usage: Refer to example file in tts_infer/
```
from tts_infer.tts import TextToMel, MelToWav
from tts_infer.transliterate import XlitEngine
from tts_infer.num_to_word_on_sent import normalize_nums

import re
from scipy.io.wavfile import write

text_to_mel = TextToMel(glow_model_dir='/path/to/glow-tts/checkpoint/dir', device='cuda')
mel_to_wav = MelToWav(hifi_model_dir='/path/to/hifi/checkpoint/dir', device='cuda')

def translit(text, lang):
    reg = re.compile(r'[a-zA-Z]')
    engine = XlitEngine(lang)
    words = [engine.translit_word(word, topk=1)[lang][0] if reg.match(word) else word for word in text.split()]
    updated_sent = ' '.join(words)
    return updated_sent
    
def run_tts(text, lang):
    text = text.replace('।', '.') # only for hindi models
    text_num_to_word = normalize_nums(text, lang) # converting numbers to words in lang
    text_num_to_word_and_transliterated = translit(text_num_to_word, lang) # transliterating english words to lang
    
    mel = text_to_mel.generate_mel(text_num_to_word_and_transliterated)
    audio, sr = mel_to_wav.generate_wav(mel)
    write(filename='temp.wav', rate=sr, data=audio) # for saving wav file, if needed
    return (sr, audio)
```


### 2. Spectogram Training (glow-tts)

filelists for glow-tts

train.txt
> abs-filepath|transcript

in case of multispeaker:
> abs-filepath|numerical_speaker_id|transcript

Note - speaker id should be zero indexed

```
cd ./scripts
bash train_glow.sh
```
### 3. Genrate Mels

```
cd ./scripts
bash generate_mels.sh
```
### 4. Vocoder Training (hifi-gan)
filelists for glow-tts

train.txt
> absolute_filepath_1


```
cd ./scripts
bash train_hifi.sh
```
### 5. Inference
```
cd ./scripts
bash infer.sh
```
