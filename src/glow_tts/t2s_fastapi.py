from starlette.responses import StreamingResponse
from texttospeech import MelToWav, TextToMel
from typing import Optional
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
import uvicorn
import base64

app = FastAPI()


class TextJson(BaseModel):
    text: str
    lang: Optional[str] = "hi"
    gender: Optional[str] = "male"


glow_hi_male = TextToMel(glow_model_dir="", device="")
glow_hi_female = TextToMel(glow_model_dir="", device="")
hifi_hi = MelToWav(hifi_model_dir="", device="")


available_choice = {
    "hi_male": [glow_hi_male, hifi_hi],
    "hi_female": [glow_hi_female, hifi_hi],
}


@app.post("/TTS/")
async def tts(input: TextJson):
    text = input.text
    lang = input.lang
    gender = input.gender

    choice = lang + "_" + gender
    if choice in available_choice.keys():
        t2s = available_choice[choice]
    else:
        raise HTTPException(
            status_code=400, detail={"error": "Requested model not found"}
        )

    if text:
        mel = t2s[0].generate_mel(text)
        data, sr = t2s[1].generate_wav(mel)
        t2s.save_audio("out.wav", data, sr)
    else:
        raise HTTPException(status_code=400, detail={"error": "No text"})

    ## to return outpur as a file
    # audio = open('out.wav', mode='rb')
    # return StreamingResponse(audio, media_type="audio/wav")

    with open("out.wav", "rb") as audio_file:
        encoded_bytes = base64.b64encode(audio_file.read())
        encoded_string = encoded_bytes.decode()
    return {"encoding": "base64", "data": encoded_string, "sr": sr}


if __name__ == "__main__":
    uvicorn.run(
        "t2s_fastapi:app", host="127.0.0.1", port=5000, log_level="info", reload=True
    )
