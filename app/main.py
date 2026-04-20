"""
app/main.py
-----------
Point d'entrée de l'API FastAPI.
Expose 4 endpoints :
  GET  /health          → état du service
  GET  /model/info      → infos sur le modèle chargé
  POST /predict         → prédiction unique
  POST /predict/batch   → prédictions multiples
"""

from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.model import load_model, predict, get_bundle, MODEL_VERSION
from app.schemas import (
    SonarInput,
    SonarPrediction,
    ModelInfo,
    HealthCheck,
)


# ── Lifespan : chargement du modèle au démarrage ───────────────────────────────
# C'est le pattern moderne FastAPI (remplace @app.on_event("startup"))
@asynccontextmanager
async def lifespan(app: FastAPI):
    # ----- au démarrage -----
    print("🚀 Démarrage de l'API — chargement du modèle...")
    try:
        load_model()
        print("✅ API prête.")
    except FileNotFoundError as e:
        print(f"⚠️  {e}")
    yield
    # ----- à l'arrêt -----
    print("🛑 Arrêt de l'API.")


# ── Application ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Sonar ML API",
    description=(
        "API de classification sonar : distingue une Mine (M) d'un Rocher (R) "
        "à partir de 60 mesures d'énergie sonar, via un modèle SVM entraîné "
        "sur le dataset UCI Sonar."
    ),
    version=MODEL_VERSION,
    lifespan=lifespan,
    docs_url="/docs",     # Swagger UI
    redoc_url="/redoc",   # ReDoc
)

# CORS — autorise tous les origines (à restreindre en production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ══════════════════════════════════════════════════════════════════════════════
#  ENDPOINT 1 — Health check
# ══════════════════════════════════════════════════════════════════════════════
@app.get(
    "/health",
    response_model=HealthCheck,
    tags=["Monitoring"],
    summary="Vérifie que l'API est vivante et le modèle chargé",
)
def health():
    try:
        get_bundle()
        model_loaded = True
        status = "ok"
    except RuntimeError:
        model_loaded = False
        status = "degraded"

    return HealthCheck(
        status=status,
        model_loaded=model_loaded,
        version=MODEL_VERSION,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  ENDPOINT 2 — Informations sur le modèle
# ══════════════════════════════════════════════════════════════════════════════
@app.get(
    "/model/info",
    response_model=ModelInfo,
    tags=["Modèle"],
    summary="Retourne les métadonnées du modèle actuellement chargé",
)
def model_info():
    try:
        bundle = get_bundle()
    except RuntimeError:
        raise HTTPException(status_code=503, detail="Modèle non chargé.")

    return ModelInfo(
        model_type   =type(bundle["model"]).__name__,
        best_params  =bundle["best_params"],
        cv_score     =bundle["cv_score"],
        model_version=MODEL_VERSION,
        status       ="loaded",
    )


# ══════════════════════════════════════════════════════════════════════════════
#  ENDPOINT 3 — Prédiction unique
# ══════════════════════════════════════════════════════════════════════════════
@app.post(
    "/predict",
    response_model=SonarPrediction,
    tags=["Prédiction"],
    summary="Prédit si un signal sonar correspond à une Mine ou un Rocher",
)
def predict_single(payload: SonarInput):
    """
    Envoie 60 valeurs numériques (énergie sonar, entre 0.0 et 1.0)
    et reçois la prédiction : **M** (Mine) ou **R** (Rock)
    avec les probabilités associées.
    """
    try:
        result = predict(payload.features)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur de prédiction : {e}")

    return SonarPrediction(
        **result,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


# ══════════════════════════════════════════════════════════════════════════════
#  ENDPOINT 4 — Prédiction batch (plusieurs signaux d'un coup)
# ══════════════════════════════════════════════════════════════════════════════
@app.post(
    "/predict/batch",
    response_model=List[SonarPrediction],
    tags=["Prédiction"],
    summary="Prédit sur plusieurs signaux sonar en une seule requête",
)
def predict_batch(payloads: List[SonarInput]):
    """
    Envoie une liste de signaux sonar, reçois une liste de prédictions.
    Utile pour traiter plusieurs mesures d'un coup.
    """
    if len(payloads) > 100:
        raise HTTPException(
            status_code=400,
            detail="Maximum 100 signaux par requête batch."
        )

    results = []
    ts = datetime.now(timezone.utc).isoformat()

    for payload in payloads:
        try:
            result = predict(payload.features)
            results.append(SonarPrediction(**result, timestamp=ts))
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    return results


# ── Point d'entrée local (développement) ──────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
