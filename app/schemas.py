from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator


# 
#  Request Schemas (Input Validation)
# 

class SonarInput(BaseModel):
    """
    Single sonar signal input for prediction.

    Represents 60 energy measurements from different sonar frequencies.
    Each measurement should be normalized to the [0.0, 1.0] range
    (as per the UCI Sonar dataset preprocessing).

    Attributes:
        features: List of exactly 60 float values representing sonar energy levels.
                  Each value must be in the range [0.0, 1.0].

    Example:
        {
            "features": [0.02, 0.03, 0.04, ..., 0.01]  # 60 values
        }
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
        """
        Validate that all feature values are within the acceptable range.

        Args:
            features: List of feature values to validate.

        Returns:
            list[float]: The validated features list.

        Raises:
            ValueError: If any feature value is outside [0.0, 1.0].
        """
        for idx, value in enumerate(features):
            if not (0.0 <= value <= 1.0):
                raise ValueError(
                    f"Feature at index {idx} has value {value}, "
                    f"but must be in range [0.0, 1.0]"
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
    """
    Single prediction result from the sonar classifier.

    Contains the predicted class, confidence scores, and metadata
    about the prediction.

    Attributes:
        prediction: Predicted class label ("M" for Mine, "R" for Rock).
        label_full: Human-readable class name ("Mine" or "Rock").
        probability_mine: Confidence score for Mine class [0.0, 1.0].
        probability_rock: Confidence score for Rock class [0.0, 1.0].
        confidence: Maximum of the two probability scores.
        model_version: Version identifier of the model used.
        timestamp: ISO 8601 formatted UTC timestamp of prediction time.
    """

    prediction: Literal["M", "R"] = Field(
        ...,
        description="Predicted class: 'M' (Mine) or 'R' (Rock)"
    )
    label_full: Literal["Mine", "Rock"] = Field(
        ...,
        description="Human-readable class label"
    )
    probability_mine: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Probability of being a Mine"
    )
    probability_rock: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Probability of being a Rock"
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Maximum probability (confidence level)"
    )
    model_version: str = Field(
        ...,
        description="Version of the model used for prediction"
    )
    timestamp: str = Field(
        ...,
        description="ISO 8601 UTC timestamp of prediction"
    )


class ModelInfo(BaseModel):
    """
    Metadata about the currently loaded SVM model.

    Provides information about the model's configuration, performance,
    and availability status.

    Attributes:
        model_type: Classifier type (e.g., "SVC" for Support Vector Classifier).
        best_params: Dictionary of optimal hyperparameters from GridSearchCV.
        cv_score: Best cross-validation accuracy score [0.0, 1.0].
        model_version: Version identifier of the model.
        status: Current model availability status.
    """

    model_type: str = Field(
        ...,
        description="Type of classifier (e.g., SVC)"
    )
    best_params: dict = Field(
        ...,
        description="Optimal hyperparameters from grid search"
    )
    cv_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Best cross-validation accuracy score"
    )
    model_version: str = Field(
        ...,
        description="Version identifier of the model"
    )
    status: Literal["loaded", "not_loaded"] = Field(
        ...,
        description="Model availability status"
    )


class HealthCheck(BaseModel):
    """
    API health status response.

    Indicates the overall health of the API and whether
    the ML model is properly loaded and ready for inference.

    Attributes:
        status: Overall API status ("ok" or "degraded").
        model_loaded: Whether the model is successfully loaded.
        version: Version of the API.
    """

    status: Literal["ok", "degraded"] = Field(
        ...,
        description="Overall API health status"
    )
    model_loaded: bool = Field(
        ...,
        description="Whether the ML model is loaded and ready"
    )
    version: str = Field(
        ...,
        description="API version identifier"
    )

