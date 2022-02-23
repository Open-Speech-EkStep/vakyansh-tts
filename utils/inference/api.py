from starlette.responses import StreamingResponse
from tts import MelToWav, TextToMel
from advanced_tts import load_all_models, run_tts_paragraph
from typing import Optional
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
import uvicorn
import base64
import argparse
import json
import time

app = FastAPI()


class TextJson(BaseModel):
    text: str
    lang: Optional[str] = "hi"
    noise_scale: Optional[float]=0.667
    length_scale: Optional[float]=1.0
    transliteration: Optional[int]=1
    number_conversion: Optional[int]=1
    split_sentences: Optional[int]=1
    wav_file: str



@app.post("/TTS/")
async def tts(input: TextJson):
    text = input.text
    lang = input.lang

    args = Namespace(**(json.loads(input)))

    args.wav_file = '../../results/'+time.time() + '.wav'

    if text:
        sr, audio = run_tts_paragraph(args)
    else:
        raise HTTPException(status_code=400, detail={"error": "No text"})

    ## to return outpur as a file
    # audio = open('out.wav', mode='rb')
    # return StreamingResponse(audio, media_type="audio/wav")

    with open(args.wav_file, "rb") as audio_file:
        encoded_bytes = base64.b64encode(audio_file.read())
        encoded_string = encoded_bytes.decode()
    return {"encoding": "base64", "data": encoded_string, "sr": sr}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--acoustic", required=True, type=str)
    parser.add_argument("-v", "--vocoder", required=True, type=str)
    parser.add_argument("-d", "--device", type=str, default="cpu")
    parser.add_argument("-L", "--lang", type=str, required=True)

    args = parser.parse_args()

    load_all_models(args)

    uvicorn.run(
        "t2s_fastapi:app", host="127.0.0.1", port=5000, log_level="info", reload=True
    )
