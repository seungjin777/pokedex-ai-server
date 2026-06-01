from fastapi import FastAPI, UploadFile
from transformers import ViTForImageClassification, ViTImageProcessor
from PIL import Image
import torch
import torch.nn.functional as F
import io
import logging
import os
from fastapi.responses import FileResponse


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    result = "success" if confidence >= 0.4 else "fail"

    logger.info(f"number: {number}, confidence: {confidence:.2%}, result: {result}")

    return {
        "number": number,
        "confidence": confidence,
        "result": result
    }


# usdz 파일이 저장된 폴더 경로
MODELS_DIR = "/Users/seung/Documents/pokemon_project/pokemon_models/usdzs"

@app.get("/pokemon/model/{pokemon_id}")
async def get_pokemon_model(pokemon_id: int):
    file_path = os.path.join(MODELS_DIR, f"{pokemon_id}.usdz")
    
    if not os.path.exists(file_path):
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="모델 파일을 찾을 수 없습니다")
    
    return FileResponse(
        file_path,
        media_type="model/vnd.usdz+zip",
        filename=f"{pokemon_id}.usdz"
    )