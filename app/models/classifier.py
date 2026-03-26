import os
import time
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import f1_score, mean_squared_error
from sklearn.feature_extraction.text import TfidfVectorizer

class TextClassifier:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=1000)
        self.model = MultinomialNB()
        self.model_path = Path("models_saved")
        self.model_path.mkdir(exist_ok=True)
        
    def train(self, csv_path):
        """Entraîne le modèle sur le dataset CSV"""
        try:
            # Charger les données
            df = pd.read_csv(csv_path)
            X = df['text'].values
            y = df['category'].values
            # Transformer les labels en nombres
            self.classes = np.unique(y)
            self.label_map = {label: idx for idx, label in enumerate(self.classes)}
            y_encoded = np.array([self.label_map[label] for label in y])
            # Vectoriser et entraîner
            X_vectorized = self.vectorizer.fit_transform(X)
            self.model.fit(X_vectorized, y_encoded)
            # Sauvegarder le modèle
            joblib.dump(self.vectorizer, self.model_path / 'vectorizer.pkl')
            joblib.dump(self.model, self.model_path / 'model.pkl')
            joblib.dump(self.label_map, self.model_path / 'label_map.pkl')
            return {"status": "success", "classes": list(self.classes)}
        except Exception as e:
            raise Exception(f"Erreur lors de l'entraînement: {str(e)}")
    
    def load_model(self):
        """Charge un modèle pré-entraîné"""
        try:
            if (self.model_path / 'model.pkl').exists():
                self.vectorizer = joblib.load(self.model_path / 'vectorizer.pkl')
                self.model = joblib.load(self.model_path / 'model.pkl')
                self.label_map = joblib.load(self.model_path / 'label_map.pkl')
                self.classes = list(self.label_map.keys())
                return True
            return False
        except Exception as e:
            print(f"Erreur lors du chargement du modèle: {e}")
            return False
    
    def predict(self, text):
        """Prédit la catégorie d'un texte"""
        try:
            if not hasattr(self.model, 'predict'):
                if not self.load_model():
                    raise Exception("Modèle non entraîné")
            X_vectorized = self.vectorizer.transform([text])
            prediction = self.model.predict(X_vectorized)[0]
            # Trouver le label correspondant
            for label, idx in self.label_map.items():
                if idx == prediction:
                    return label
            return "inconnu"
        except Exception as e:
            raise Exception(f"Erreur lors de la prédiction: {str(e)}")
    
    def evaluate(self, csv_path):
        """Évalue le modèle avec F1 score et RMSE"""
        try:
            if not self.load_model():
                raise Exception("Modèle non entraîné")
            
            df = pd.read_csv(csv_path)
            X = df['text'].values
            y_true = df['category'].values
            # Prédire
            predictions = [self.predict(text) for text in X]
            # Calculer F1 Score
            f1 = f1_score(y_true, predictions, average='weighted')
            # Calculer RMSE (encodage numérique)
            y_true_encoded = [self.label_map[label] for label in y_true]
            y_pred_encoded = [self.label_map[pred] for pred in predictions]
            rmse = np.sqrt(mean_squared_error(y_true_encoded, y_pred_encoded))
            return {
                "f1_score": float(f1),
                "rmse": float(rmse),
                "total_samples": len(X),
                "predictions": dict(list(zip(range(min(5, len(X))), predictions[:5])))
            }
        except Exception as e:
            raise Exception(f"Erreur lors de l'évaluation: {str(e)}")
