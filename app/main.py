import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.model import load_model, predict, get_bundle, MODEL_VERSION
from app.schemas import SonarInput, SonarPrediction, ModelInfo, HealthCheck

# 
#  Logging Configuration
# 

logger = logging.getLogger(__name__)

# 
#  Application Lifespan & Startup/Shutdown Logic
# 


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage application lifecycle events (startup and shutdown).

    This context manager is invoked by FastAPI:
    - On startup: Loads the pre-trained model bundle into memory
    - On shutdown: Gracefully cleans up resources

    Args:
        app: FastAPI application instance.

    Yields:
        None
    """
    # ─────────────────────── STARTUP ───────────────────────────
    logger.info("🚀 Starting Sonar ML API - Loading model...")
    try:
        load_model()
        logger.info("✅ API is ready for inference.")
    except FileNotFoundError as e:
        logger.error(f"⚠️  Startup warning: {e}")
    except Exception as e:
        logger.critical(f"Critical startup error: {e}", exc_info=True)
        raise

    yield

    # ─────────────────────── SHUTDOWN ──────────────────────────
    logger.info("🛑 Shutting down Sonar ML API.")


# 
#  FastAPI Application Setup
# 

app = FastAPI(
    title="Sonar Classification API",
    description=(
        "Machine Learning API for sonar signal classification. "
        "Using an SVM classifier trained on the UCI Sonar dataset, "
        "this service distinguishes between Mine (M) and Rock (R) signals "
        "based on 60 frequency-based energy measurements."
    ),
    version=MODEL_VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add CORS middleware for cross-origin requests
#     In production, restrict to specific trusted origins.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# 
#  ENDPOINT 1: Health Check
# 

@app.get(
    "/health",
    response_model=HealthCheck,
    tags=["Monitoring"],
    summary="Service health and model status",
    description="Check if the API is running and the ML model is loaded.",
)
def health() -> HealthCheck:
    """
    Health check endpoint for monitoring and diagnostics.

    Verifies:
    - API responsiveness
    - Model availability in memory
    - Overall service status

    Returns:
        HealthCheck: Status object with model availability info.
    """
    try:
        get_bundle()
        model_loaded = True
        status = "ok"
    except RuntimeError:
        model_loaded = False
        status = "degraded"

    logger.debug(f"Health check: status={status}, model_loaded={model_loaded}")

    return HealthCheck(
        status=status,
        model_loaded=model_loaded,
        version=MODEL_VERSION,
    )


# 
#  ENDPOINT 2: Model Metadata
# 

@app.get(
    "/model/info",
    response_model=ModelInfo,
    tags=["Model"],
    summary="Retrieve model configuration and metadata",
    description="Get information about the currently loaded SVM classifier.",
)
def model_info() -> ModelInfo:
    """
    Get metadata about the currently loaded model.

    Returns information including:
    - Model type (e.g., SVC)
    - Hyperparameter configuration
    - Cross-validation performance score
    - Model version

    Returns:
        ModelInfo: Metadata dictionary.

    Raises:
        HTTPException: 503 if model is not loaded.
    """
    try:
        bundle = get_bundle()
    except RuntimeError as e:
        logger.error(f"Model info request failed: {e}")
        raise HTTPException(status_code=503, detail="Model not loaded.")

    return ModelInfo(
        model_type=type(bundle["model"]).__name__,
        best_params=bundle["best_params"],
        cv_score=bundle["cv_score"],
        model_version=MODEL_VERSION,
        status="loaded",
    )


# 
#  ENDPOINT 3: Single Prediction
# 

@app.post(
    "/predict",
    response_model=SonarPrediction,
    tags=["Prediction"],
    summary="Classify a single sonar signal",
    description="Predict whether a sonar signal represents a Mine or Rock.",
)
def predict_single(payload: SonarInput) -> SonarPrediction:
    """
    Perform single-sample inference on a sonar signal.

    Accepts 60 normalized energy measurements and returns
    a classification prediction with confidence scores.

    Args:
        payload: SonarInput containing 60 feature values.

    Returns:
        SonarPrediction: Prediction result with probabilities.

    Raises:
        HTTPException: 503 if model not loaded, 500 on inference error.
    """
    try:
        result = predict(payload.features)
    except RuntimeError as e:
        logger.error(f"Prediction failed - model issue: {e}")
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(f"Prediction failed - inference error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")

    result["timestamp"] = datetime.now(timezone.utc).isoformat()

    logger.debug(f"Prediction completed: {result['prediction']}")

    return SonarPrediction(**result)


# 
#  ENDPOINT 4: Batch Predictions
# 

@app.post(
    "/predict/batch",
    response_model=List[SonarPrediction],
    tags=["Prediction"],
    summary="Classify multiple sonar signals",
    description="Perform batch inference on multiple sonar signals in a single request.",
)
def predict_batch(payloads: List[SonarInput]) -> List[SonarPrediction]:
    """
    Perform batch inference on multiple sonar signals.

    Processes a list of signals and returns predictions for each.
    Useful for high-throughput scenarios.

    Args:
        payloads: List of SonarInput objects.

    Returns:
        List[SonarPrediction]: List of prediction results.

    Raises:
        HTTPException: 400 if batch size exceeds limit,
                      503 if model not loaded,
                      500 on inference error.
    """
    # Enforce batch size limit to prevent resource exhaustion
    MAX_BATCH_SIZE = 100
    if len(payloads) > MAX_BATCH_SIZE:
        logger.warning(f"Batch request exceeds limit: {len(payloads)} > {MAX_BATCH_SIZE}")
        raise HTTPException(
            status_code=400,
            detail=f"Maximum batch size is {MAX_BATCH_SIZE} signals."
        )

    timestamp = datetime.now(timezone.utc).isoformat()
    results = []

    for idx, payload in enumerate(payloads):
        try:
            result = predict(payload.features)
            result["timestamp"] = timestamp
            results.append(SonarPrediction(**result))
        except Exception as e:
            logger.error(f"Batch inference failed at index {idx}: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    logger.info(f"Batch prediction completed: {len(results)} samples processed")

    return results


# 
#  Local Development Entry Point
# 

if __name__ == "__main__":
    import uvicorn

    logger.info("Starting development server...")
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )

