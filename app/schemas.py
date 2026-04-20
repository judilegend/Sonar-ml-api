"""
app/schemas.py
--------------
Définit la forme exacte des données que l'API accepte (input)
et renvoie (output). Pydantic valide automatiquement les types.
"""

from pydantic import BaseModel, Field, field_validator
from typing import Literal
from datetime import datetime


class SonarInput(BaseModel):
    """
    Payload d'entrée : 60 valeurs numériques entre 0 et 1
    représentant les énergies du signal sonar par fréquence.
    """
    features: list[float] = Field(
        ...,
        min_length=60,
        max_length=60,
        description="60 valeurs numériques (énergie sonar par fréquence)"
    )

    @field_validator("features")
    @classmethod
    def check_range(cls, v):
        for i, val in enumerate(v):
            if not (0.0 <= val <= 1.0):
                raise ValueError(
                    f"La feature [{i}] = {val} est hors plage [0.0, 1.0]"
                )
        return v

    model_config = {
        "json_schema_extra": {
            "example": {
                "features": [0.02] * 60  # exemple minimaliste
            }
        }
    }


class SonarPrediction(BaseModel):
    """
    Réponse de l'API après prédiction.
    """
    prediction       : Literal["M", "R"]           # Mine ou Rock
    label_full       : Literal["Mine", "Rock"]      # version lisible
    probability_mine : float                         # proba d'être une Mine
    probability_rock : float                         # proba d'être un Rocher
    confidence       : float                         # max des deux probas
    model_version    : str                           # version du modèle chargé
    timestamp        : str                           # heure de la prédiction


class ModelInfo(BaseModel):
    """
    Informations sur le modèle actuellement chargé.
    """
    model_type   : str
    best_params  : dict
    cv_score     : float
    model_version: str
    status       : Literal["loaded", "not_loaded"]


class HealthCheck(BaseModel):
    status       : Literal["ok", "degraded"]
    model_loaded : bool
    version      : str
