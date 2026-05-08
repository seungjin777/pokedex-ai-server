from fastapi import FastAPI, UploadFile
from transformers import ViTForImageClassification, ViTImageProcessor
from PIL import Image
import torch
import torch.nn.functional as F
import io

app = FastAPI()

model_id = "skshmjn/Pokemon-classifier-gen9-1025"
print("모델 로드 중...")
model = ViTForImageClassification.from_pretrained(model_id)
processor = ViTImageProcessor.from_pretrained(model_id)
model.eval()
print("모델 로드 완료")

@app.get("/")
async def root():
    return {"message": "Hello pokedex"}

@app.post("/pokemon/predictions")
async def prediction_pokemon(file: UploadFile):
    img_bytes = await file.read()
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    inputs = processor(images=img, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    probabilities = F.softmax(outputs.logits, dim=-1)
    confidence = probabilities.max().item()

    predicted_id = outputs.logits.argmax(-1).item()
    number = predicted_id + 1
    result = "success" if confidence >= 0.5 else "fail"

    return {
        "number": number,
        "confidence": confidence,
        "result": result
    }