# ══════════════════════════════════════════════════════════════
#  STAGE 1 — TRAINER
#  Installe les dépendances et entraîne le modèle.
#  Ce stage produit models/sonar_model.pkl
# ══════════════════════════════════════════════════════════════
FROM python:3.11-slim AS trainer

WORKDIR /app

# Copier les dépendances en premier (cache Docker optimal)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copier le code source
COPY ml/    ml/
COPY app/   app/

# Entraîner le modèle → génère models/sonar_model.pkl
RUN python ml/train.py


# ══════════════════════════════════════════════════════════════
#  STAGE 2 — API (image finale légère)
#  Copie uniquement ce dont l'API a besoin depuis le stage 1.
#  L'image finale ne contient PAS le code d'entraînement.
# ══════════════════════════════════════════════════════════════
FROM python:3.11-slim AS api

# Utilisateur non-root pour la sécurité
RUN useradd --create-home --shell /bin/bash appuser

WORKDIR /app

# Copier les dépendances
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copier le code de l'API
COPY app/ app/

# Copier le modèle produit par le stage trainer
COPY --from=trainer /app/models/ models/

# Changer le propriétaire des fichiers
RUN chown -R appuser:appuser /app

USER appuser

# Port exposé
EXPOSE 8000

# Variables d'environnement par défaut
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Commande de démarrage
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
