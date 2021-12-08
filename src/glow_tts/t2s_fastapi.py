from texttospeech import TextToSpeech
from typing import Optional
import wavio
from io import BytesIO
from fastapi import FastAPI, Response, HTTPException
import uvicorn

app = FastAPI()

def get_audio(data, sr):
    audio = BytesIO()
    wavio.write(audio, data, sr, sampwidth=2)
    audio.seek(0)
    return audio.read()

@app.get("/predict/")
async def inference(text: str, noise: Optional[float]=0.667, length: Optional[float]=1.0):
    if text:
        data, sr = t2s.generate_audio(text, noise_scale=noise, length_scale=length)
        audio = get_audio(data, sr)
    else:
        raise HTTPException(status_code=400, detail="Bad Request")

    return Response(audio, media_type="audio/wav")

t2s = TextToSpeech(
            glow_model_dir='',
            hifi_model_dir='',
            device='')

if __name__ == '__main__':
    uvicorn.run(
        "t2s_fastapi:app", 
        host="127.0.0.1", 
        port=5000, 
        log_level="info",
        reload=True)
    
