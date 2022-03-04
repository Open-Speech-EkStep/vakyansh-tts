import gradio as gr
import argparse
import numpy as np
from argparse import Namespace
from advanced_tts import load_all_models, run_tts_paragraph


def hit_tts(textbox, slider_noise_scale, slider_length_sclae, choice_transliteration, choice_number_conversion, choice_split_sentences):
    inputs_to_gradio = {'text' : textbox,
                        'noise_scale': slider_noise_scale,
                        'length_scale': slider_length_sclae,
                        'transliteration' : 1 if choice_transliteration else 0,
                        'number_conversion' : 1 if choice_number_conversion else 0,
                        'split_sentences' : 1 if choice_split_sentences else 0
                        }

    args = Namespace(**inputs_to_gradio)
    args.wav = None
    args.lang = lang

    if args.text:
        sr, audio = run_tts_paragraph(args)
        return (sr, audio)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--acoustic", required=True, type=str)
    parser.add_argument("-v", "--vocoder", required=True, type=str)
    parser.add_argument("-d", "--device", type=str, default="cpu")
    parser.add_argument("-L", "--lang", type=str, required=True)

    global lang

    args = parser.parse_args()    
    lang = args.lang
    load_all_models(args)
    
    textbox = gr.inputs.Textbox(placeholder="Enter Text to run", default="", label="TTS")
    slider_noise_scale = gr.inputs.Slider(minimum=0, maximum=1.0, step=0.001, default=0.667, label='Enter Noise Scale')
    slider_length_sclae = gr.inputs.Slider(minimum=0, maximum=2.0, step=0.1, default=1.0, label='Enter Slider Scale')

    choice_transliteration = gr.inputs.Checkbox(default=True, label="Transliteration")
    choice_number_conversion = gr.inputs.Checkbox(default=True, label="Number Conversion")
    choice_split_sentences = gr.inputs.Checkbox(default=True, label="Split Sentences")

        
   
    op = gr.outputs.Audio(type="numpy", label=None)

    inputs_to_gradio = [textbox, slider_noise_scale, slider_length_sclae, choice_transliteration, choice_number_conversion, choice_split_sentences]
    iface = gr.Interface(fn=hit_tts, inputs=inputs_to_gradio, outputs=op, theme='huggingface', title='Run TTS example')
    iface.launch(share=True, enable_queue=True)