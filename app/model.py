"""
app/model.py
------------
Responsable du chargement du bundle (model + scaler + metadata)
et de l'inférence. Séparé de main.py pour la testabilité.
"""

import pickle
import numpy as np
from pathlib import Path

# Chemin vers le bundle sauvegardé par ml/train.py
MODEL_PATH = Path("models/sonar_model.pkl")
MODEL_VERSION = "1.0.0"

# Variable globale — chargée une seule fois au démarrage de l'API
_bundle = None


def load_model() -> dict:
    """
    Charge le bundle pickle depuis le disque.
    Appelé au démarrage de FastAPI (lifespan event).
    Lève une exception si le fichier est absent.
    """
    global _bundle

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Modèle introuvable : {MODEL_PATH}\n"
            "Lance d'abord : python ml/train.py"
        )

    with open(MODEL_PATH, "rb") as f:
        _bundle = pickle.load(f)

    print(f"✅ Modèle chargé — params : {_bundle['best_params']}")
    print(f"   CV score : {_bundle['cv_score']}")
    return _bundle


def get_bundle() -> dict:
    """Retourne le bundle chargé. Lève une erreur si non initialisé."""
    if _bundle is None:
        raise RuntimeError("Le modèle n'est pas encore chargé.")
    return _bundle


def predict(features: list[float]) -> dict:
    """
    Prend 60 valeurs brutes, les normalise avec le scaler,
    et retourne la prédiction + les probabilités.

    Retourne un dict compatible avec SonarPrediction.
    """
    bundle = get_bundle()
    model  = bundle["model"]
    scaler = bundle["scaler"]
    label_map = bundle["label_map"]   # {0: 'M', 1: 'R'} ou inverse

    # Reshape → (1, 60) comme attendu par sklearn
    X = np.array(features).reshape(1, -1)

    # Normalisation avec le MÊME scaler que l'entraînement
    X_scaled = scaler.transform(X)

    # Prédiction de la classe + probabilités
    pred_class = int(model.predict(X_scaled)[0])
    probas     = model.predict_proba(X_scaled)[0]  # [proba_classe_0, proba_classe_1]

    # label_map : ex {0: 'M', 1: 'R'}
    # On cherche quel index correspond à M et à R
    idx_m = [k for k, v in label_map.items() if v == "M"][0]
    idx_r = [k for k, v in label_map.items() if v == "R"][0]

    prob_mine = round(float(probas[idx_m]), 4)
    prob_rock = round(float(probas[idx_r]), 4)
    label     = label_map[pred_class]   # "M" ou "R"

    return {
        "prediction"      : label,
        "label_full"      : "Mine" if label == "M" else "Rock",
        "probability_mine": prob_mine,
        "probability_rock": prob_rock,
        "confidence"      : round(max(prob_mine, prob_rock), 4),
        "model_version"   : MODEL_VERSION,
    }
