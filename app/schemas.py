from datetime import datetime
from typing import Literal
from pydantic import BaseModel, Field, field_validator, ConfigDict

# 
#  Request Schemas (Input Validation)
# 

class SonarInput(BaseModel):
    """
    Single sonar signal input for prediction.
    Represents 60 energy measurements from different sonar frequencies.
    """
    
    features: list[float] = Field(
        ...,
        min_length=60,
        max_length=60,
        description="Array of 60 sonar frequency energy measurements normalized to [0.0, 1.0]"
    )

    @field_validator("features")
    @classmethod
    def validate_feature_range(cls, features: list[float]) -> list[float]:
        """Validate that all feature values are within [0.0, 1.0]."""
        for idx, value in enumerate(features):
            if not (0.0 <= value <= 1.0):
                raise ValueError(
                    f"Feature at index {idx} has value {value}, but must be in range [0.0, 1.0]"
                )
        return features

    model_config = {
        "json_schema_extra": {
            "example": {
                "features": [0.02] * 60
            }
        }
    }


# 
#  Response Schemas (Output Serialization)
# 

class SonarPrediction(BaseModel):
    """Result of a sonar classification."""
    
    # Disable Pydantic's protected namespace for fields starting with 'model_'
    model_config = ConfigDict(protected_namespaces=())

    prediction: Literal["M", "R"] = Field(..., description="Predicted class: 'M' (Mine) or 'R' (Rock)")
    label_full: Literal["Mine", "Rock"] = Field(..., description="Human-readable class label")
    probability_mine: float = Field(..., ge=0.0, le=1.0)
    probability_rock: float = Field(..., ge=0.0, le=1.0)
    confidence: float = Field(..., ge=0.0, le=1.0)
    model_version: str = Field(..., description="Version of the model used")
    timestamp: str = Field(..., description="ISO 8601 UTC timestamp of prediction")


class ModelInfo(BaseModel):
    """Metadata about the currently loaded SVM model."""
    
    model_config = ConfigDict(protected_namespaces=())

    model_type: str = Field(..., description="Classifier type (e.g., SVC)")
    best_params: dict = Field(..., description="Optimal hyperparameters from GridSearchCV")
    cv_score: float = Field(..., ge=0.0, le=1.0)
    model_version: str = Field(..., description="Model version identifier")
    status: Literal["loaded", "not_loaded"] = Field(..., description="Current availability status")


class HealthCheck(BaseModel):
    """API and Model health status."""
    
    model_config = ConfigDict(protected_namespaces=())

    status: Literal["ok", "degraded"] = Field(..., description="Overall API health")
    model_loaded: bool = Field(..., description="True if the .pkl model is ready")
    version: str = Field(..., description="API version")