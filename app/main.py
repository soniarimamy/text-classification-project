import uvicorn
from pathlib import Path
from fastapi import FastAPI, HTTPException, UploadFile, File
app = FastAPI(title="Text Classification API")

BASE_DIR=Path('/app')
DATA_DIR = BASE_DIR  / "data"
TEMP_DIR = BASE_DIR / "temp_files"
MODELS_DIR = BASE_DIR / "models_saved"
DATA_DIR.mkdir(exist_ok=True)
TEMP_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

@app.get('/')
async def root():
    return {"message": "Text Classification API", "endpoints": ["/train", "/classify", "/evaluate"]}
@app.post('/train')
async def train():
    return {"code": 201, "msg": "service for training", "data": []}
@app.post('/classify')
async def classify():
    return {"code": 201, "msg": "service for text classification",
        "data": []}

@app.post('/evaluate')
async def evaluate():
    return {"code": 201, "msg": "service for model evluation",
        "data": []}

uvicorn.run(app, host="0.0.0.0", port=8000)
