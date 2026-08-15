"""
BlueFish AI - Fisherman Advisory Agent (Production-Ready)
======================================================
The primary conversational interface for individual fishermen. Orchestrates
Models 1, 2, 3, 7, 8, 9 to produce one coherent recommendation.

Design principles applied:
  - Group A models (1, 2, 3) read from cache, NEVER computed live.
  - Group C models (7, 9) run on-demand per request.
  - Model 8 always runs (pure astronomical math).
  - Strict Feature Contract enforcement for ONNX and Scikit-Learn models.
  - Graceful degradation: one model failing doesn't crash the request.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import onnxruntime as ort
import pickle

logger = logging.getLogger("bluefish.advisory_agent")
logger.setLevel(logging.INFO)

# ======================================================================
# Result containers
# ======================================================================

@dataclass
class ModelResult:
    model_name: str
    ok: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    latency_ms: float = 0.0

@dataclass
class Recommendation:
    vessel_id: str
    date: str
    fishing_ground: Optional[ModelResult] = None
    fronts_eddies: Optional[ModelResult] = None
    migration_forecast: Optional[ModelResult] = None
    route: Optional[ModelResult] = None
    time_window: Optional[ModelResult] = None
    vessel_segment: Optional[ModelResult] = None
    degraded: bool = False
    degraded_reasons: List[str] = field(default_factory=list)

# ======================================================================
# Cache Interface
# ======================================================================

class PredictionCache:
    def get(self, key: str) -> Optional[Dict[str, Any]]: raise NotImplementedError
    def set(self, key: str, value: Dict[str, Any], ttl_seconds: int = 90000) -> None: raise NotImplementedError

class InMemoryPredictionCache(PredictionCache):
    def __init__(self): self._store: Dict[str, Dict[str, Any]] = {}
    def get(self, key: str) -> Optional[Dict[str, Any]]: return self._store.get(key)
    def set(self, key: str, value: Dict[str, Any], ttl_seconds: int = 90000) -> None: self._store[key] = value

# ======================================================================
# Model 1 - ONNX Runtime wrapper (Strict Feature Contract)
# ======================================================================

class Model1Client:
    def __init__(self, stage1_path: str, stage2_path: str, feature_order: List[str]):
        self.stage1_session = ort.InferenceSession(stage1_path, providers=["CPUExecutionProvider"])
        self.stage2_session = ort.InferenceSession(stage2_path, providers=["CPUExecutionProvider"])
        self.feature_order = feature_order

        # CRITICAL: Verify feature count matches exactly what the ONNX model expects
        expected_shape = self.stage1_session.get_inputs()[0].shape
        expected_n_features = expected_shape[-1] if isinstance(expected_shape[-1], int) else None
        if expected_n_features is not None and expected_n_features != len(feature_order):
            raise ValueError(
                f"Model 1 ONNX expects {expected_n_features} features, "
                f"but feature_order provided has {len(feature_order)} entries. "
                f"Refusing to run inference with a mismatched feature contract."
            )

        self.stage1_input_name = self.stage1_session.get_inputs()[0].name
        self.stage2_input_name = self.stage2_session.get_inputs()[0].name

    def predict(self, feature_row: Dict[str, float], presence_threshold: float = 0.5) -> Dict[str, Any]:
        missing = [f for f in self.feature_order if f not in feature_row]
        if missing:
            raise ValueError(f"Missing required features for Model 1: {missing}")

        x = np.array([[feature_row[f] for f in self.feature_order]], dtype=np.float32)

        stage1_out = self.stage1_session.run(None, {self.stage1_input_name: x})
        presence_prob = self._extract_positive_class_prob(stage1_out)

        result: Dict[str, Any] = {"presence_probability": float(presence_prob)}

        if presence_prob >= presence_threshold:
            stage2_out = self.stage2_session.run(None, {self.stage2_input_name: x})
            log_intensity = float(np.asarray(stage2_out[0]).flatten()[0])
            intensity_hours = float(np.expm1(log_intensity))
            result["predicted_intensity_hours"] = max(0.0, intensity_hours)
        else:
            result["predicted_intensity_hours"] = None

        return result

    @staticmethod
    def _extract_positive_class_prob(stage1_out) -> float:
        probs = stage1_out[1] if len(stage1_out) > 1 else stage1_out[0]
        arr = np.asarray(probs)
        if arr.dtype == object:
            first = probs[0]
            if isinstance(first, dict):
                return float(first.get(1, list(first.values())[-1]))
        arr = arr.flatten()
        return float(arr[-1])

# ======================================================================
# Model 9 - KMeans (Strict Feature Count Validation)
# ======================================================================

class Model9Client:
    def __init__(self, kmeans_path: str, scaler_path: Optional[str] = None):
        with open(kmeans_path, "rb") as f:
            self.kmeans = pickle.load(f)
        self.scaler = None
        if scaler_path:
            with open(scaler_path, "rb") as f:
                self.scaler = pickle.load(f)
                
        # Validate expected features
        self.expected_features = getattr(self.kmeans, "n_features_in_", None)

    def predict(self, vessel_features: List[float]) -> Dict[str, Any]:
        if self.expected_features is not None and len(vessel_features) != self.expected_features:
            raise ValueError(
                f"Model 9 expects {self.expected_features} features, got {len(vessel_features)}."
            )
            
        x = np.array([vessel_features], dtype=np.float64)
        if self.scaler is not None:
            x = self.scaler.transform(x)
        cluster = int(self.kmeans.predict(x)[0])
        return {"segment_id": cluster}

# ======================================================================
# THE AGENT
# ======================================================================

class FishermanAdvisoryAgent:
    def __init__(
        self,
        cache: PredictionCache,
        model1_client: Optional[Model1Client] = None,
        model2_service=None,
        model7_service=None,
        model8_service=None,
        model9_client: Optional[Model9Client] = None,
    ):
        self.cache = cache
        self.model1 = model1_client
        self.model2 = model2_service
        self.model7 = model7_service
        self.model8 = model8_service
        self.model9 = model9_client

    def _safe_call(self, model_name: str, fn, *args, **kwargs) -> ModelResult:
        start = time.monotonic()
        try:
            data = fn(*args, **kwargs)
            latency = (time.monotonic() - start) * 1000
            logger.info(f"model_call_success model={model_name} latency_ms={latency:.1f}")
            return ModelResult(model_name=model_name, ok=True, data=data, latency_ms=latency)
        except Exception as e:
            latency = (time.monotonic() - start) * 1000
            logger.error(f"model_call_failed model={model_name} error={e} latency_ms={latency:.1f}")
            return ModelResult(model_name=model_name, ok=False, error=str(e), latency_ms=latency)

    # Group A Cache Lookups
    def _get_cached_fishing_ground(self, date: str, zone_key: str) -> ModelResult:
        def _lookup():
            cached = self.cache.get(f"model1:{date}:{zone_key}")
            if cached is None: raise LookupError("No cached Model 1 prediction yet.")
            return cached
        return self._safe_call("model1_fishing_ground", _lookup)

    def _get_cached_fronts_eddies(self, date: str, zone_key: str) -> ModelResult:
        def _lookup():
            cached = self.cache.get(f"model2:{date}:{zone_key}")
            if cached is None: raise LookupError("No cached Model 2 output yet.")
            return cached
        return self._safe_call("model2_fronts_eddies", _lookup)

    def _get_cached_migration_forecast(self, date: str, zone_key: str) -> ModelResult:
        def _lookup():
            cached = self.cache.get(f"model3:{date}:{zone_key}")
            if cached is None: raise LookupError("No cached Model 3 forecast yet.")
            return cached
        return self._safe_call("model3_migration_forecast", _lookup)

    # Group C On-Demand
    def _get_route(self, start_lat, start_lon, target_lat, target_lon) -> ModelResult:
        if not self.model7: return ModelResult("model7_route", False, error="Not configured")
        return self._safe_call("model7_route", self.model7.predict, start_lat, start_lon, target_lat, target_lon)

    def _get_time_window(self, date: str, lat: float, lon: float) -> ModelResult:
        if not self.model8: return ModelResult("model8_time_window", False, error="Not configured")
        return self._safe_call("model8_time_window", self.model8.predict, date, lat, lon)

    def _get_vessel_segment(self, vessel_features: List[float]) -> ModelResult:
        if not self.model9: return ModelResult("model9_segment", False, error="Not configured")
        return self._safe_call("model9_segment", self.model9.predict, vessel_features)

    # Main Entry Point
    def get_recommendation(
        self,
        vessel_id: str,
        date: str,
        current_lat: float,
        current_lon: float,
        target_zone_lat: Optional[float] = None,
        target_zone_lon: Optional[float] = None,
        zone_key: Optional[str] = None,
        vessel_features: Optional[List[float]] = None,
    ) -> Recommendation:
        rec = Recommendation(vessel_id=vessel_id, date=date)

        if zone_key is None:
            rec.degraded = True
            rec.degraded_reasons.append("No zone_key provided - Group A predictions skipped.")
        else:
            rec.fishing_ground = self._get_cached_fishing_ground(date, zone_key)
            rec.fronts_eddies = self._get_cached_fronts_eddies(date, zone_key)
            rec.migration_forecast = self._get_cached_migration_forecast(date, zone_key)

        query_lat = target_zone_lat if target_zone_lat is not None else current_lat
        query_lon = target_zone_lon if target_zone_lon is not None else current_lon
        rec.time_window = self._get_time_window(date, query_lat, query_lon)

        if target_zone_lat is not None and target_zone_lon is not None:
            rec.route = self._get_route(current_lat, current_lon, target_zone_lat, target_zone_lon)
        else:
            rec.route = ModelResult("model7_route", False, error="No target zone specified")

        if vessel_features is not None:
            rec.vessel_segment = self._get_vessel_segment(vessel_features)

        for result in [rec.fishing_ground, rec.fronts_eddies, rec.migration_forecast, rec.route, rec.time_window, rec.vessel_segment]:
            if result is not None and not result.ok and result.error:
                if "not configured" not in result.error and "No target zone specified" not in result.error:
                    rec.degraded = True
                    rec.degraded_reasons.append(f"{result.model_name}: {result.error}")

        return rec

    # ------------------------------------------------------------------
    # UPGRADED: Enhanced LLM Context Generator
    # Formats the raw data into a clean, natural-language friendly dict
    # so the LLM doesn't hallucinate math or misinterpret probabilities.
    # ------------------------------------------------------------------
    @staticmethod
    def to_llm_context(rec: Recommendation) -> Dict[str, Any]:
        def extract(result: Optional[ModelResult]):
            return result.data if (result is not None and result.ok) else None

        fg = extract(rec.fishing_ground)
        tw = extract(rec.time_window)
        rt = extract(rec.route)

        # Format into human-readable strings for the LLM prompt
        fg_str = "Unknown"
        if fg and "presence_probability" in fg:
            prob = fg["presence_probability"]
            fg_str = f"{prob*100:.0f}% probability of fish."
            if fg.get("predicted_intensity_hours") is not None:
                fg_str += f" Estimated fishing time: {fg['predicted_intensity_hours']:.1f} hours."

        tw_str = "Unknown"
        if tw and "feeding_windows" in tw:
            windows = [f"{w['type']} at {w['peak_time']} ({w['reason']})" for w in tw["feeding_windows"]]
            tw_str = "; ".join(windows)

        rt_str = "Not calculated"
        if rt and "route" in rt:
            rt_str = f"{rt['steps']} waypoints calculated."

        return {
            "vessel_id": rec.vessel_id,
            "date": rec.date,
            "recommendation_summary": {
                "fishing_ground_status": fg_str,
                "optimal_fishing_times": tw_str,
                "route_status": rt_str
            },
            "degraded": rec.degraded,
            "degraded_reasons": rec.degraded_reasons,
            "raw_data": {
                "fishing_ground": fg,
                "fronts_and_eddies": extract(rec.fronts_eddies),
                "migration_forecast": extract(rec.migration_forecast),
                "route": rt,
                "time_window": tw,
                "vessel_segment": extract(rec.vessel_segment),
            }
        }

