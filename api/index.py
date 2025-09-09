from http.cookies import SimpleCookie
from PIL import Image
import torch, io, numpy as np
from transformers import AutoImageProcessor, AutoModel
from fastapi import FastAPI, UploadFile, Response
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# CORS 全开，方便调试
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

device = "cuda" if torch.cuda.is_available() else "cpu"
processor = AutoImageProcessor.from_pretrained("facebook/dinov2-small")
model = AutoModel.from_pretrained("facebook/dinov2-small").to(device).eval()

@app.post("/dinov2")
async def dinov2(file: UploadFile):
    img = Image.open(io.BytesIO(await file.read())).convert("RGB")
    inputs = processor(images=img, return_tensors="pt").to(device)
    with torch.no_grad():
        feat = model(**inputs).last_hidden_state[:, 0].cpu().numpy().squeeze()
    return {"feat": feat.tolist()}

# Vercel 入口点
async def app(scope, receive, send):
    await app(scope, receive, send)