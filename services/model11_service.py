"""
BlueFish AI - Model 11 Service (XGBoost Climate Risk)
======================================================
Wraps the XGBoost climate change impact model and its scaler.
Predicts long-term fishing viability given climate scenario features.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger("bluefish.model11_service")


class ClimateRiskService:
    """
    Wraps the XGBoost_Boss_Model.pkl for climate risk inference.

    The model expects a feature vector describing regional ocean/climate
    conditions and predicts a risk score or impact category.
    """

    def __init__(self, xgb_model: Any, scaler: Any):
        self.model = xgb_model
        self.scaler = scaler
        logger.info("✓ Model 11 (Climate Risk XGBoost) service initialized.")

    def predict(self, features: List[float]) -> Dict[str, Any]:
        """
        Runs climate risk prediction.

        Args:
            features: Feature vector matching the training schema of XGBoost_Boss_Model.

        Returns:
            Dict with 'risk_score', 'risk_category', and 'raw_prediction'.
        """
        try:
            x = np.array([features], dtype=np.float64)
            x_scaled = self.scaler.transform(x)
            raw_pred = self.model.predict(x_scaled)
            score = float(raw_pred[0])

            # Categorize risk
            if score < 0.3:
                category = "LOW"
            elif score < 0.6:
                category = "MODERATE"
            elif score < 0.8:
                category = "HIGH"
            else:
                category = "SEVERE"

            return {
                "risk_score": round(score, 4),
                "risk_category": category,
                "raw_prediction": float(score),
            }
        except Exception as e:
            logger.error(f"Model 11 inference failed: {e}")
            raise

    def predict_proba(self, features: List[float]) -> Dict[str, Any]:
        """Returns class probabilities if the model supports predict_proba."""
        try:
            x = np.array([features], dtype=np.float64)
            x_scaled = self.scaler.transform(x)
            probas = self.model.predict_proba(x_scaled)
            return {"probabilities": probas[0].tolist()}
        except AttributeError:
            # Regression model — predict_proba not available
            return self.predict(features)
        except Exception as e:
            logger.error(f"Model 11 predict_proba failed: {e}")
            raise
