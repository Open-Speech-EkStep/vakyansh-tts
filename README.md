# vakyansh-tts
Text to Speech for Indic languages

### 1. Installation and SetUp

Clone repo
```
git clone https://github.com/Open-Speech-EkStep/vakyansh-tts
```
Build conda virtual environment
```
cd ./vakyansh-tts
conda create --name <env_name> python=3.7
conda activate <env_name>
pip install -r requirement.txt
```
Install [apex](https://github.com/NVIDIA/apex); commit: 37cdaf4 for Mixed-precision training
```
cd ..
git clone https://github.com/NVIDIA/apex
cd apex
git checkout 37cdaf4
pip install -v --disable-pip-version-check --no-cache-dir ./
cd ../vakyansh-tts
```
Build Monotonic Alignment Search Code (Cython)
```
cd src/glow_tts/monotonic_align/
python setup.py build_ext --inplace
cd ../../../
```

### 2. Spectogram Training (glow-tts)

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

```
cd ./scripts
bash train_hifi.sh
```
### 4. Inference
```
cd ./scripts
bash infer.sh
```
