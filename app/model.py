import logging
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# 
#  Configuration
#

MODEL_PATH = Path("models/sonar_model.pkl")
MODEL_VERSION = "1.0.0"

# Module-level cache: initialized once during API startup
_bundle: Dict | None = None

logger = logging.getLogger(__name__)


# 
#  Model Loading
# 
def load_model() -> Dict:
    """
    Load the model bundle from disk into module-level cache.

    This function is called once during API startup (lifespan event).
    Subsequent requests retrieve the cached model via get_bundle().

    Returns:
        Dict: The loaded model bundle containing:
            - "model": Trained SVC classifier
            - "scaler": Fitted StandardScaler
            - "label_map": Dict mapping class indices to labels
            - "best_params": Dict of hyperparameters
            - "cv_score": Float cross-validation score

    Raises:
        FileNotFoundError: If the model file does not exist.
        pickle.UnpicklingError: If the file cannot be deserialized.
    """
    global _bundle

    if not MODEL_PATH.exists():
        error_msg = (
            f"Model file not found: {MODEL_PATH}\n"
            f"Please run: python ml/train.py"
        )
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)

    try:
        with open(MODEL_PATH, "rb") as f:
            _bundle = pickle.load(f)

        logger.info(
            f"✅ Model loaded successfully | Params: {_bundle['best_params']}"
        )
        logger.info(f"   Cross-validation score: {_bundle['cv_score']}")

        return _bundle

    except Exception as e:
        logger.error(f"Failed to load model bundle: {e}")
        raise


def get_bundle() -> Dict:
    """
    Retrieve the cached model bundle.

    Returns:
        Dict: The global model bundle.

    Raises:
        RuntimeError: If the model has not been loaded yet.
    """
    if _bundle is None:
        error_msg = "Model has not been loaded. Please start the API first."
        logger.error(error_msg)
        raise RuntimeError(error_msg)

    return _bundle


# 
#  Inference
# 

def predict(features: List[float]) -> Dict[str, float | str]:
    """
    Perform inference on a single sonar signal.

    This function:
    1. Loads the model bundle
    2. Validates and reshapes input features
    3. Applies the same scaling used during training
    4. Generates predictions and class probabilities
    5. Maps predictions back to original class labels

    Args:
        features: List of 60 float values representing sonar energy levels.
                  Should be in the range [0.0, 1.0].

    Returns:
        Dict containing:
            - "prediction": Predicted class label ("M" or "R")
            - "label_full": Human-readable label ("Mine" or "Rock")
            - "probability_mine": Probability of being a Mine [0.0, 1.0]
            - "probability_rock": Probability of being a Rock [0.0, 1.0]
            - "confidence": Maximum of the two probabilities
            - "model_version": Version of the loaded model

    Raises:
        RuntimeError: If the model is not loaded.
        ValueError: If feature dimensions are incorrect.
    """
    bundle = get_bundle()
    model = bundle["model"]
    scaler = bundle["scaler"]
    label_map = bundle["label_map"]

    # Reshape to (1, 60) for sklearn compatibility
    features_array = np.array(features, dtype=np.float32).reshape(1, -1)

    # Apply the training scaler (critical for feature normalization)
    features_scaled = scaler.transform(features_array)

    # Generate prediction and probabilities
    predicted_class_idx = int(model.predict(features_scaled)[0])
    probabilities = model.predict_proba(features_scaled)[0]

    # Map indices to original class labels
    mine_idx = _find_class_index(label_map, "M")
    rock_idx = _find_class_index(label_map, "R")

    prob_mine = round(float(probabilities[mine_idx]), 4)
    prob_rock = round(float(probabilities[rock_idx]), 4)
    predicted_label = label_map[predicted_class_idx]

    logger.debug(
        f"Prediction: {predicted_label} | "
        f"Mine: {prob_mine:.4f} | Rock: {prob_rock:.4f}"
    )

    return {
        "prediction": predicted_label,
        "label_full": "Mine" if predicted_label == "M" else "Rock",
        "probability_mine": prob_mine,
        "probability_rock": prob_rock,
        "confidence": round(max(prob_mine, prob_rock), 4),
        "model_version": MODEL_VERSION,
    }


# 
#  Utility Functions
# 

def _find_class_index(label_map: Dict[int, str], target_label: str) -> int:
    """
    Find the array index corresponding to a target class label.

    Args:
        label_map: Dictionary mapping indices to class labels.
        target_label: Target label ("M" or "R").

    Returns:
        int: Index corresponding to target_label.

    Raises:
        ValueError: If target_label is not found in label_map.
    """
    for idx, label in label_map.items():
        if label == target_label:
            return idx

    raise ValueError(f"Label '{target_label}' not found in label_map: {label_map}")

