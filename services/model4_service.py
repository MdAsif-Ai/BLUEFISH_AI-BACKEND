"""
BlueFish AI - Model 4 Service (TFT Seasonal Forecast)
======================================================
Wraps the pytorch_forecasting Temporal Fusion Transformer checkpoint.
This is marked OPTIONAL — the dependency is heavy and not required for
core fleet/safety functionality.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger("bluefish.model4_service")


class TFTForecastService:
    """
    Service wrapper for the Temporal Fusion Transformer seasonal forecast model.

    Requires: pytorch_forecasting, lightning
    Install: pip install pytorch_forecasting lightning

    Note: pytorch_forecasting models are loaded from .ckpt files via
    the TemporalFusionTransformer.load_from_checkpoint() class method.
    """

    def __init__(self, checkpoint_path: str):
        try:
            from pytorch_forecasting import TemporalFusionTransformer
            logger.info(f"Loading TFT checkpoint from {checkpoint_path}...")
            self.model = TemporalFusionTransformer.load_from_checkpoint(checkpoint_path)
            self.model.eval()
            logger.info("✓ TFT model ready.")
        except ImportError:
            raise ImportError(
                "pytorch_forecasting is not installed. "
                "Install it with: pip install pytorch_forecasting lightning"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load TFT checkpoint: {e}")

    def predict(self, encoder_data: Any, decoder_data: Any) -> Dict[str, Any]:
        """
        Runs seasonal forecast.

        In production, encoder_data and decoder_data should be PyTorch DataLoaders
        or the TimeSeriesDataSet-formatted tensors expected by the TFT model.

        Returns a forecast dict with 'median', 'q10', 'q90' predictions.
        """
        try:
            import torch
            with torch.no_grad():
                predictions = self.model.predict(
                    encoder_data,
                    return_x=False,
                    mode="quantiles",
                )
            return {
                "predictions": predictions.cpu().numpy().tolist(),
                "quantiles": [0.1, 0.5, 0.9],
            }
        except Exception as e:
            logger.error(f"TFT inference failed: {e}")
            raise

    def predict_simple(self, features: List[float], horizon_weeks: int = 12) -> Dict[str, Any]:
        """
        Simplified inference path for single-location seasonal outlook.
        Returns a placeholder forecast when full DataLoader integration is not available.
        """
        logger.warning("TFT predict_simple: returning placeholder forecast — full DataLoader integration required.")
        return {
            "horizon_weeks": horizon_weeks,
            "seasonal_outlook": "moderate",
            "note": "Full TFT integration requires proper TimeSeriesDataSet pipeline.",
        }
