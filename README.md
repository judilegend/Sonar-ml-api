# 🔊 Sonar ML API

API de classification sonar basée sur SVM — distingue une **Mine** d'un **Rocher**
à partir de 60 mesures d'énergie sonar (dataset UCI Sonar).

## Architecture

```
sonar-ml-api/
├── app/
│   ├── main.py        ← FastAPI : endpoints, routing
│   ├── model.py       ← chargement du modèle + inférence
│   └── schemas.py     ← validation Pydantic (input/output)
├── ml/
│   └── train.py       ← entraînement SVM + sauvegarde du modèle
├── tests/
│   └── test_api.py    ← tests pytest
├── models/            ← généré par train.py (gitignore)
│   └── sonar_model.pkl
├── Dockerfile         ← multi-stage : trainer + api
├── docker-compose.yml
├── .github/workflows/
│   └── ci_cd.yml      ← GitHub Actions CI/CD
└── requirements.txt
```

## Démarrage rapide

### Option A — En local (sans Docker)

```bash
# 1. Installer les dépendances
pip install -r requirements.txt

# 2. Entraîner le modèle (génère models/sonar_model.pkl)
python ml/train.py

# 3. Lancer l'API
uvicorn app.main:app --reload

# 4. Ouvrir la documentation interactive
# http://localhost:8000/docs
```

### Option B — Avec Docker Compose

```bash
# Build + lancement de l'API
docker compose up --build

# Entraînement du modèle dans Docker (si nécessaire)
docker compose run --rm trainer

# L'API est disponible sur http://localhost:8000
```

## Endpoints

| Méthode | URL              | Description                        |
|---------|------------------|------------------------------------|
| GET     | `/health`        | État de l'API et du modèle         |
| GET     | `/model/info`    | Métadonnées du modèle chargé       |
| POST    | `/predict`       | Prédiction sur un signal           |
| POST    | `/predict/batch` | Prédictions sur plusieurs signaux  |
| GET     | `/docs`          | Documentation Swagger UI           |

## Exemple de requête

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [0.02, 0.03, 0.04, 0.02, 0.01, 0.03, 0.05, 0.02,
                    0.03, 0.04, 0.02, 0.01, 0.03, 0.05, 0.02, 0.03,
                    0.04, 0.02, 0.01, 0.03, 0.05, 0.02, 0.03, 0.04,
                    0.02, 0.01, 0.03, 0.05, 0.02, 0.03, 0.04, 0.02,
                    0.01, 0.03, 0.05, 0.02, 0.03, 0.04, 0.02, 0.01,
                    0.03, 0.05, 0.02, 0.03, 0.04, 0.02, 0.01, 0.03,
                    0.05, 0.02, 0.03, 0.04, 0.02, 0.01, 0.03, 0.05,
                    0.02, 0.03, 0.04, 0.01]}'
```

Réponse :
```json
{
  "prediction": "R",
  "label_full": "Rock",
  "probability_mine": 0.1823,
  "probability_rock": 0.8177,
  "confidence": 0.8177,
  "model_version": "1.0.0",
  "timestamp": "2024-01-15T10:30:00+00:00"
}
```

## Tests

```bash
pytest tests/ -v
```

## Pipeline CI/CD

Le fichier `.github/workflows/ci_cd.yml` définit 3 jobs :

1. **Test** — lint flake8 + pytest (sur chaque push et PR)
2. **Build** — build Docker multi-stage (sur push main)
3. **Publish** — push vers Docker Hub (sur push main)

### Configurer les secrets GitHub

Dans ton repo → Settings → Secrets and variables → Actions :

- `DOCKERHUB_USERNAME` → ton username Docker Hub
- `DOCKERHUB_TOKEN` → [créer ici](https://hub.docker.com/settings/security)
