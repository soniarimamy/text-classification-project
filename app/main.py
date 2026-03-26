import uuid
import os
import io
from typing import Optional
import uvicorn
import pandas as pds
from pathlib import Path
from pydantic import BaseModel
from .models.classifier import TextClassifier
from fastapi import FastAPI, HTTPException, UploadFile, File
app = FastAPI(title="Text Classification API")
classifier = TextClassifier()

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
TEMP_DIR = BASE_DIR / "temp_files"
MODELS_DIR = BASE_DIR / "models_saved"

DATA_DIR.mkdir(exist_ok=True, parents=True)
TEMP_DIR.mkdir(exist_ok=True, parents=True)
MODELS_DIR.mkdir(exist_ok=True, parents=True)

@app.get('/')
async def root():
    return {"message": "Text Classification API", "endpoints": ["/train", "/classify", "/evaluate"]}

class TextInput(BaseModel):
    text: str

class TrainResponse(BaseModel):
    status: str
    classes: list
    message: Optional[str] = None

class EvaluateResponse(BaseModel):
    f1_score: float
    rmse: float
    total_samples: int
    predictions: dict

class ClassifyResponse(BaseModel):
    category: str
    text: str

def save_temp_file(content: bytes, suffix: str = ".csv") -> str:
    """Sauvegarde un fichier temporaire avec un nom unique"""
    temp_id = uuid.uuid4().hex
    temp_path = TEMP_DIR / f"temp_{temp_id}{suffix}"
    with open(temp_path, 'wb') as f:
        f.write(content)
    return str(temp_path)

def cleanup_temp_file(file_path: str):
    """Supprime un fichier temporaire"""
    try:
        if file_path and os.path.exists(file_path):
            os.remove(file_path)
    except Exception as e:
        print(f"Warning: Could not delete temp file {file_path}: {e}")

@app.post('/train')
async def train(file: UploadFile = File(...)):
    temp_file_path = None
    try:
        contents = await file.read()
        if not contents:
            raise HTTPException(400, "Fichier vide")
        print("Fichier recu: name=", file.filename, "taille: ", len(contents), " bytes")
        temp_file_path = save_temp_file(contents)
        print("Fichier temporairement sauvegardé dans", temp_file_path)
        df = pds.read_csv(temp_file_path)
        if 'text' not in df.columns or 'category' not in df.columns:
            raise HTTPException(400, "CSV doit contenir les collones text et category")
        result=classifier.train(temp_file_path)
        print("Entraintement terminé, resultat=", result)
        return TrainResponse(**result, message="Modele entrainé avec succes")
    except pds.errors.EmptyDataError as e:
        raise HTTPException(400, "Fichier CSV vide ou invalide")
    except Exception as e:
        print("Erreur lors de l'entrainement", str(e))
        raise HTTPException(500, "Erreur lors de l'entrainement", str(e))
    finally:
        if temp_file_path:
            cleanup_temp_file(temp_file_path)

@app.post("/classify", response_model=ClassifyResponse)
async def classify(text_input: TextInput):
    """Endpoint pour classifier un texte"""
    try:
        if not classifier.load_model():
            raise HTTPException(400, "Modèle non entraîné. Veuillez d'abord entraîner le modèle.")
        category = classifier.predict(text_input.text)
        return ClassifyResponse(category=category, text=text_input.text)
    except Exception as e:
        raise HTTPException(500, f"Erreur lors de la classification: {str(e)}")
@app.post("/evaluate", response_model=EvaluateResponse)
async def evaluate(file: UploadFile = File(...)):
    """Endpoint pour évaluer le modèle"""
    temp_file_path = None
    try:
        if not classifier.load_model():
            raise HTTPException(400, "Modèle non entraîné. Veuillez d'abord entraîner le modèle.")
        # Lire et sauvegarder le fichier
        contents = await file.read()
        temp_file_path = save_temp_file(contents)
        # Lire le CSV pour validation
        df = pds.read_csv(temp_file_path)
        # Évaluer le modèle
        metrics = classifier.evaluate(temp_file_path)
        return EvaluateResponse(**metrics)
    except Exception as e:
        raise HTTPException(500, f"Erreur lors de l'évaluation: {str(e)}")
    finally:
        # Nettoyer le fichier temporaire
        if temp_file_path:
            cleanup_temp_file(temp_file_path)
