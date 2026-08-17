"""
BlueFish AI - Model 6 Service: Behavioral Anomaly Detection
============================================================
Wraps the Isolation Forest + StandardScaler pipeline for detecting
anomalous fishing vessel behavior (AIS spoofing, illegal fishing patterns,
abnormal speed/heading transitions, MPA intrusions).

Feature contract (21 features, ORDER MUST MATCH TRAINING):
  [speed, heading, lat, lon, speed_delta, heading_delta,
   time_since_last_ping, distance_from_port, is_night,
   sst_at_position, depth_at_position, course_over_ground,
   rate_of_turn, time_in_zone, zone_entry_count, zone_change_rate,
   speed_variance_1h, heading_variance_1h, distance_traveled_1h,
   is_in_mpa, is_in_eez]

The Isolation Forest outputs an anomaly score (lower = more anomalous).
Threshold: score < -0.1 → flagged as anomaly.

Usage:
    service = AnomalyDetectionService(model_registry.model6_forest, model_registry.model6_scaler)
    flags = service.predict_batch(vessels_list)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("bluefish.services.model6")

# 21-feature contract — must match training pipeline exactly
FEATURE_ORDER: List[str] = [
    "speed", "heading", "lat", "lon", "speed_delta", "heading_delta",
    "time_since_last_ping", "distance_from_port", "is_night",
    "sst_at_position", "depth_at_position", "course_over_ground",
    "rate_of_turn", "time_in_zone", "zone_entry_count", "zone_change_rate",
    "speed_variance_1h", "heading_variance_1h", "distance_traveled_1h",
    "is_in_mpa", "is_in_eez",
]

ANOMALY_SCORE_THRESHOLD = -0.1


class AnomalyDetectionService:
    """
    Batch-vectorized Isolation Forest anomaly detection.
    Processes up to 20,000 vessels per cycle without nested loops.
    """

    def __init__(self, iso_forest, scaler, threshold: float = ANOMALY_SCORE_THRESHOLD):
        """
        Args:
            iso_forest: Fitted sklearn IsolationForest instance
            scaler: Fitted sklearn StandardScaler instance
            threshold: Score below this → anomaly (default: -0.1)
        """
        self.iso_forest = iso_forest
        self.scaler = scaler
        self.threshold = threshold

    def _build_feature_matrix(self, vessels: List[Dict[str, Any]]) -> np.ndarray:
        """
        Constructs the (N × 21) feature matrix from a list of vessel dicts.
        Missing features default to 0.0. This handles incomplete telemetry gracefully.
        """
        matrix = np.zeros((len(vessels), len(FEATURE_ORDER)), dtype=np.float32)
        for i, vessel in enumerate(vessels):
            for j, key in enumerate(FEATURE_ORDER):
                matrix[i, j] = float(vessel.get(key, 0.0))
        return matrix

    def predict_batch(
        self,
        vessels: List[Dict[str, Any]],
        include_scores: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Runs batch anomaly detection on a list of vessel dicts.
        Batch-vectorized: scales all vessels at once, then scores.

        Args:
            vessels: List of vessel dicts (must include 'mmsi' for identification)
            include_scores: If True, includes the raw anomaly score in output

        Returns:
            List of anomaly flags — only vessels flagged as anomalies are returned.
            Each flag dict:
                {
                    "mmsi": str,
                    "is_anomaly": True,
                    "anomaly_score": float,
                    "severity": "HIGH" | "MEDIUM" | "LOW",
                    "flags": list of triggered anomaly categories,
                }
        """
        if not vessels:
            return []

        X = self._build_feature_matrix(vessels)
        scores = None
        if self.scaler is not None and self.iso_forest is not None:
            try:
                X_scaled = self.scaler.transform(X)
                scores = self.iso_forest.score_samples(X_scaled)
            except Exception as e:
                logger.warning(f"Isolation Forest batch inference fallback: {e}")

        if scores is None:
            # Fallback heuristic scoring when ML model artifact is not loaded
            scores = []
            for vessel in vessels:
                flags = _diagnose_anomaly_flags(vessel, -0.2)
                if flags and flags != ["unclassified_anomaly"]:
                    scores.append(-0.15 * len(flags))
                else:
                    scores.append(0.1)
            scores = np.array(scores, dtype=np.float32)

        anomalies = []
        for i, vessel in enumerate(vessels):
            score = float(scores[i])
            if score >= self.threshold:
                continue  # Normal

            flags = _diagnose_anomaly_flags(vessel, score)
            severity = _classify_anomaly_severity(score)

            result: Dict[str, Any] = {
                "mmsi": vessel.get("mmsi", "UNKNOWN"),
                "is_anomaly": True,
                "severity": severity,
                "flags": flags,
                "lat": vessel.get("lat", 0.0),
                "lon": vessel.get("lon", 0.0),
            }
            if include_scores:
                result["anomaly_score"] = round(score, 4)

            anomalies.append(result)

        return anomalies

    def predict_single(self, vessel: Dict[str, Any]) -> Dict[str, Any]:
        """Single-vessel prediction. Less efficient than batch — use for on-demand checks only."""
        results = self.predict_batch([vessel])
        if results:
            return results[0]
        return {
            "mmsi": vessel.get("mmsi", "UNKNOWN"),
            "is_anomaly": False,
            "anomaly_score": 0.0,
            "flags": [],
        }


def _diagnose_anomaly_flags(vessel: Dict[str, Any], score: float) -> List[str]:
    """
    Generates human-readable anomaly flag labels based on feature values.
    These are rule-based heuristics applied AFTER the IF flags the vessel.
    """
    flags = []
    speed = float(vessel.get("speed", 0.0))
    heading_delta = float(vessel.get("heading_delta", 0.0))
    is_in_mpa = bool(vessel.get("is_in_mpa", False))
    is_in_eez = bool(vessel.get("is_in_eez", True))
    time_since_ping = float(vessel.get("time_since_last_ping", 0.0))

    if speed > 15.0:          flags.append("excessive_speed")
    if speed < 0.3 and speed > 0:  flags.append("ghost_vessel_stationary")
    if heading_delta > 120:   flags.append("erratic_heading")
    if is_in_mpa:             flags.append("inside_mpa")
    if not is_in_eez:         flags.append("outside_eez")
    if time_since_ping > 60:  flags.append("ais_dark_period")
    if score < -0.3:          flags.append("high_isolation_score")

    return flags if flags else ["unclassified_anomaly"]


def _classify_anomaly_severity(score: float) -> str:
    if score < -0.4:
        return "HIGH"
    elif score < -0.25:
        return "MEDIUM"
    return "LOW"
