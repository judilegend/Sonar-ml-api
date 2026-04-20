# ==============================================================
#  STAGE 1 — TRAINER
#  Installs dependencies and trains the model.
#  This stage produces: models/sonar_model.pkl
# ==============================================================
FROM python:3.11-slim AS trainer

WORKDIR /app

# Copy dependencies first to optimize Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY ml/  ml/
COPY app/ app/

# Train the model during the image build process
# This ensures the .pkl file is generated before moving to Stage 2
RUN python ml/train.py

# Default command for the trainer stage (if run standalone)
CMD ["echo", "Training stage complete"]


# ==============================================================
#  STAGE 2 — API (Final lightweight image)
#  Only copies what the API needs from Stage 1.
#  The final image DOES NOT contain training code or raw data.
# ==============================================================
FROM python:3.11-slim AS api

# Create a non-root user for security
RUN useradd --create-home --shell /bin/bash appuser

WORKDIR /app

# Install runtime dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy API application code
COPY app/ app/

# Copy the trained model file from the trainer stage
COPY --from=trainer /app/models/ models/

# Change file ownership to the non-root user
RUN chown -R appuser:appuser /app

USER appuser

# Expose the API port
EXPOSE 8000

# Default environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Start the FastAPI application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]