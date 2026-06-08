from fastapi import FastAPI, Request
from fastapi.responses import Response, JSONResponse
import torchaudio as ta
import torch
import io
import traceback
from chatterbox.mtl_tts import ChatterboxMultilingualTTS
import multiprocessing

app = FastAPI()

print("Loading Chatterbox Multilingual Model...")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = ChatterboxMultilingualTTS.from_pretrained(device=device)


# voice cache fo more optimazion
voice_cache = {}
default_robot_conds = model.conds 
print("Model loaded successfully!")

@app.post("/v1/audio/stream")
async def stream_audio(request: Request):
    try:
        print("\n" + "="*50)
        print(">>> NEW REQUEST RECEIVED IN PYTHON")
        print("="*50)
        
        data = await request.json()
        
        text = data.get("input", "Bonjour")
        language = data.get("language", "fr")
        voice_ref = data.get("voice_ref", "orion.wav") 
        
        exaggeration = float(data.get("exaggeration", 0.5))
        temperature = float(data.get("temperature", 0.8))
        cfg_weight = float(data.get("cfg_weight", 0.5)) 
        
        print(f"--- Text: '{text}'")
        print(f"--- Params: Voice={voice_ref} | Exag={exaggeration} | Temp={temperature} | CFG={cfg_weight}")
        
        if voice_ref and voice_ref != "default_french":
            cache_key = f"{voice_ref}_{exaggeration}"
            
            if cache_key not in voice_cache:
                print(f"[CACHE MISS] Learning voice profile for '{cache_key}' from disk...")
                model.prepare_conditionals(f"/app/voices/{voice_ref}", exaggeration=exaggeration)
                voice_cache[cache_key] = model.conds
                print("[OK] Voice learned and saved to RAM.")
            else:
                print(f"[CACHE HIT] Loading '{cache_key}' instantly from RAM.")
                
            model.conds = voice_cache[cache_key]
            print("... Generating audio ...")
            wav = model.generate(text, language_id=language, exaggeration=exaggeration, temperature=temperature, cfg_weight=cfg_weight)
            
        else:
            print("[INFO] Using generic robot voice (No cloning required).")
            model.conds = default_robot_conds
            print("... Generating audio ...")
            wav = model.generate(text, language_id=language, exaggeration=exaggeration, temperature=temperature, cfg_weight=cfg_weight)
        
        print("... Processing audio tensor ...")
        wav = wav.cpu()
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
            
        buffer = io.BytesIO()
        ta.save(buffer, wav, model.sr, format="wav")
        
        print("<<< Sending audio file back to Spring Boot.")
        print("="*50 + "\n")
        
        return Response(content=buffer.getvalue(), media_type="audio/wav")
        
    except Exception as e:
        print("\n[ERROR] EXCEPTION DURING GENERATION:")
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})