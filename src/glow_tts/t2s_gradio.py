import gradio as gr
from texttospeech import TextToSpeech


t2s = TextToSpeech(glow_model_dir='/path/to/glow-tts/checkpoint/dir', 
                    hifi_model_dir='/path/to/glow-tts/checkpoint/dir')
def run_tts(text):
    audio, sr = t2s.generate_audio(text)
    return (sr, audio)

# text = " सीआईएसएफ में उप-निरीक्षक महावीर प्रसाद गोदरा को मरणोपरांत 'शौर्य चक्र' से सम्मानित किया गया। "
# run_tts(text)

textbox = gr.inputs.Textbox(placeholder="Enter Telugu text here", default="", label="TTS")
op = gr.outputs.Audio(type="numpy", label=None)
iface = gr.Interface(fn=run_tts, inputs=textbox, outputs=op)
iface.launch(share=True)