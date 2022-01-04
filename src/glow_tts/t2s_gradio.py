import gradio as gr
from texttospeech import TextToMel, MelToWav

text_to_mel = TextToMel(
    glow_model_dir="/path/to/glow-tts/checkpoint/dir", device="cuda"
)
mel_to_wav = MelToWav(hifi_model_dir="/path/to/glow-tts/checkpoint/dir", device="cuda")


def run_tts(text):
    mel = text_to_mel.generate_mel(text)
    audio, sr = mel_to_wav.generate_wav(mel)
    return (sr, audio)


# text = " सीआईएसएफ में उप-निरीक्षक महावीर प्रसाद गोदरा को मरणोपरांत 'शौर्य चक्र' से सम्मानित किया गया। "
# run_tts(text)

textbox = gr.inputs.Textbox(
    placeholder="Enter Telugu text here", default="", label="TTS"
)
op = gr.outputs.Audio(type="numpy", label=None)
iface = gr.Interface(fn=run_tts, inputs=textbox, outputs=op)
iface.launch(share=True)
