"""
BlueFish AI - Model 1 Service: PFZ (Potential Fishing Zone) Prediction
=======================================================================
Wraps the two-stage ONNX cascade for PFZ prediction.

Stage 1 (Presence): Binary classifier — "Is there a meaningful fish presence?"
Stage 2 (Intensity): Regression — "How intense is the presence?" (0.0–1.0)

Feature contract (order MUST match ONNX training pipeline):
    [month, dayofyear, ONI_Value, sst, salinity, current_east,
     current_north, chlorophyll, current_speed, current_direction_deg]

Usage:
    service = PFZService(model_registry.model1)
    result = service.predict(features_dict)
    grid_result = service.predict_grid(grid_df)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger("bluefish.services.model1")

# Canonical feature order — must exactly match the ONNX model's training schema
FEATURE_ORDER: List[str] = [
    "month", "dayofyear", "ONI_Value", "sst", "salinity",
    "current_east", "current_north", "chlorophyll",
    "current_speed", "current_direction_deg",
]


class PFZService:
    """
    Two-stage ONNX cascade for Potential Fishing Zone prediction.
    Loaded from Supabase Storage artifacts:
      - stage1_presence.onnx
      - stage2_intensity.onnx
    """

    def __init__(self, model_tuple: Any):
        """
        Args:
            model_tuple: Can be:
                - A tuple/list (stage1_session, stage2_session) of ort.InferenceSession objects
                - A Model1Client instance (from agents.advisory_agent)
                - A single ort.InferenceSession
        """
        if hasattr(model_tuple, "stage1_session"):
            self.stage1_session = model_tuple.stage1_session
            self.stage2_session = getattr(model_tuple, "stage2_session", None)
        elif isinstance(model_tuple, (tuple, list)) and len(model_tuple) == 2:
            self.stage1_session, self.stage2_session = model_tuple
        else:
            self.stage1_session = model_tuple
            self.stage2_session = None

        self._s1_input_name: Optional[str] = None
        self._s2_input_name: Optional[str] = None

    @property
    def _stage1_input_name(self) -> str:
        if self._s1_input_name is None and self.stage1_session is not None:
            self._s1_input_name = self.stage1_session.get_inputs()[0].name
        return self._s1_input_name or "input"

    @property
    def _stage2_input_name(self) -> str:
        if self._s2_input_name is None and self.stage2_session is not None:
            self._s2_input_name = self.stage2_session.get_inputs()[0].name
        return self._s2_input_name or "input"

    def _build_feature_array(self, features: Dict[str, float]) -> np.ndarray:
        """
        Converts a feature dict to a numpy array in the correct column order.
        Missing features default to 0.0 with a warning.
        """
        row = []
        for key in FEATURE_ORDER:
            val = features.get(key)
            if val is None:
                logger.warning(f"PFZ feature '{key}' missing — defaulting to 0.0")
                val = 0.0
            row.append(float(val))
        return np.array([row], dtype=np.float32)

    def predict_point(self, features: Dict[str, float]) -> Dict[str, Any]:
        """
        Predicts PFZ probability for a single lat/lon point given feature values.

        Returns:
            {
                "presence_probability": float (0–1),
                "intensity_score": float (0–1),
                "zone_class": "HIGH" | "MEDIUM" | "LOW" | "ABSENT",
            }
        """
        if self.stage1_session is None:
            # Fallback if sessions failed to load
            return {
                "presence_probability": 0.5,
                "intensity_score": 0.5,
                "zone_class": "MEDIUM",
                "degraded": True,
            }

        X = self._build_feature_array(features)

        # Stage 1: Presence probability
        s1_output = self.stage1_session.run(None, {self._stage1_input_name: X})
        if isinstance(s1_output, (list, tuple)) and len(s1_output) > 1:
            raw_prob = s1_output[1]
            if isinstance(raw_prob, (list, np.ndarray)) and len(raw_prob) > 0:
                if isinstance(raw_prob[0], dict):
                    proba = float(raw_prob[0].get(1, 0.5))
                else:
                    proba = float(np.asarray(raw_prob).flatten()[-1])
            else:
                proba = 0.5
        else:
            proba = float(np.asarray(s1_output[0]).flatten()[-1])

        proba = float(np.clip(proba, 0.0, 1.0))

        # Stage 2: Intensity score
        intensity = 0.0
        if self.stage2_session is not None and proba > 0.4:
            try:
                s2_output = self.stage2_session.run(None, {self._stage2_input_name: X})
                raw_s2 = float(np.asarray(s2_output[0]).flatten()[0])
                intensity = float(np.clip(raw_s2, 0.0, 1.0))
            except Exception as e:
                logger.warning(f"Stage 2 intensity inference failed: {e}")
                intensity = proba

        if proba < 0.3:
            zone_class = "ABSENT"
        elif proba < 0.55:
            zone_class = "LOW"
        elif proba < 0.75:
            zone_class = "MEDIUM"
        else:
            zone_class = "HIGH"

        return {
            "presence_probability": round(proba, 4),
            "intensity_score": round(intensity, 4),
            "zone_class": zone_class,
        }

    def predict(self, features: Dict[str, float]) -> Dict[str, Any]:
        """Alias for predict_point for unified interface."""
        return self.predict_point(features)

    def predict_grid(self, grid_data: List[Dict[str, float]]) -> List[Dict[str, Any]]:
        """
        Batch prediction over a list of grid points.
        Each item in grid_data must include lat, lon, and the feature keys.
        """
        if not grid_data:
            return []

        if self.stage1_session is None:
            results = []
            for row in grid_data:
                results.append({
                    "lat": row.get("lat", 0.0),
                    "lon": row.get("lon", 0.0),
                    "presence_probability": 0.5,
                    "intensity_score": 0.5,
                    "zone_class": "MEDIUM",
                })
            return results

        X = np.array(
            [[float(row.get(k, 0.0)) for k in FEATURE_ORDER] for row in grid_data],
            dtype=np.float32,
        )

        try:
            s1_output = self.stage1_session.run(None, {self._stage1_input_name: X})
            if isinstance(s1_output, (list, tuple)) and len(s1_output) > 1:
                raw_prob = s1_output[1]
                if isinstance(raw_prob, np.ndarray) and raw_prob.ndim == 2 and raw_prob.shape[1] > 1:
                    probas = raw_prob[:, 1]
                else:
                    probas = np.asarray(s1_output[0]).flatten()
            else:
                probas = np.asarray(s1_output[0]).flatten()
        except Exception as e:
            logger.error(f"Stage 1 batch inference failed: {e}")
            probas = np.zeros(len(grid_data), dtype=np.float32)

        intensities = probas.copy()
        if self.stage2_session is not None:
            try:
                mask = probas > 0.4
                if mask.any():
                    s2_out = self.stage2_session.run(None, {self._stage2_input_name: X[mask]})
                    s2_flat = np.asarray(s2_out[0]).flatten()
                    intensities[mask] = np.clip(s2_flat, 0.0, 1.0)
            except Exception as e:
                logger.warning(f"Stage 2 batch inference failed: {e}")

        results = []
        for i, row in enumerate(grid_data):
            p = float(np.clip(probas[i], 0.0, 1.0))
            zone_class = ("ABSENT" if p < 0.3 else "LOW" if p < 0.55
                          else "MEDIUM" if p < 0.75 else "HIGH")
            results.append({
                "lat": row.get("lat", 0.0),
                "lon": row.get("lon", 0.0),
                "presence_probability": round(p, 4),
                "intensity_score": round(float(intensities[i]), 4),
                "zone_class": zone_class,
            })

        return results

    def to_geojson(self, grid_predictions: List[Dict[str, Any]], threshold: float = 0.3) -> Dict[str, Any]:
        """
        Converts grid predictions above the threshold to a GeoJSON FeatureCollection.
        """
        features = []
        for pred in grid_predictions:
            if pred.get("presence_probability", 0.0) < threshold:
                continue
            features.append({
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [pred.get("lon", 0.0), pred.get("lat", 0.0)],
                },
                "properties": {
                    "presence_probability": pred.get("presence_probability", 0.0),
                    "intensity_score": pred.get("intensity_score", 0.0),
                    "zone_class": pred.get("zone_class", "ABSENT"),
                },
            })
        return {
            "type": "FeatureCollection",
            "features": features,
            "meta": {
                "total_points": len(grid_predictions),
                "above_threshold": len(features),
                "threshold": threshold,
            },
        }
