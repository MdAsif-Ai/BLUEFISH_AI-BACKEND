"""
BlueFish AI - Model 9 Service: Vessel Segmentation (K-Means)
=============================================================
Wraps the K-Means clustering model for vessel behavioral segmentation.

Model 9 classifies fishing vessels into behavioral archetypes based on
33 trip-level features derived from historical VMS/AIS data.

Expected segments (cluster labels may vary by training run):
  - "deep_sea_trawler"   : Long trips, high fuel burn, large catch
  - "coastal_gillnetter" : Short trips, low fuel, near-coast patterns
  - "seasonal_migrator"  : Follows fish migration, variable trip length
  - "opportunistic"      : Responds to real-time PFZ alerts

Feature contract (33 features):
  Includes avg_trip_duration, avg_daily_distance, avg_fuel_per_trip,
  avg_catch_per_trip, trip_frequency_monthly, preferred_zone,
  night_fishing_ratio, monsoon_activity_ratio, and 25 more.

Usage:
    service = VesselSegmentationService(model_registry.model9_kmeans)
    result = service.predict(vessel_features_dict)
    batch_result = service.predict_batch(vessels_list)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger("bluefish.services.model9")

# 33-feature contract (order must match training pipeline)
FEATURE_ORDER: List[str] = [
    "avg_trip_duration_hours", "avg_daily_distance_km", "avg_fuel_per_trip_liters",
    "avg_catch_per_trip_kg", "trip_frequency_monthly", "preferred_lat",
    "preferred_lon", "night_fishing_ratio", "monsoon_activity_ratio",
    "speed_p25", "speed_p50", "speed_p75", "speed_std",
    "heading_entropy", "zone_diversity_index", "avg_time_in_mpa_hours",
    "mpa_entry_count", "eez_boundary_crossings", "ais_dark_hours_ratio",
    "avg_depth_at_fishing", "avg_sst_at_fishing", "catch_per_fuel_ratio",
    "days_since_last_trip", "total_trips_lifetime", "max_distance_from_port_km",
    "avg_return_time_hours", "weather_delay_ratio", "formation_fishing_ratio",
    "avg_trip_profitability", "seasonal_preference_q1", "seasonal_preference_q2",
    "seasonal_preference_q3", "seasonal_preference_q4",
]

# Human-readable segment labels (mapped from cluster IDs post-training)
SEGMENT_LABELS: Dict[int, str] = {
    0: "coastal_gillnetter",
    1: "deep_sea_trawler",
    2: "seasonal_migrator",
    3: "opportunistic",
    4: "artisanal_inshore",
    5: "offshore_longliner",
}


class VesselSegmentationService:
    """K-Means vessel behavioral segmentation service."""

    def __init__(self, kmeans_model):
        """
        Args:
            kmeans_model: Fitted sklearn KMeans or pickle-loaded equivalent
        """
        self.model = kmeans_model
        self._n_clusters: int = getattr(kmeans_model, "n_clusters", len(SEGMENT_LABELS))

    def _build_feature_vector(self, vessel: Dict[str, Any]) -> np.ndarray:
        """Converts vessel dict to feature vector. Missing features → 0.0."""
        return np.array(
            [float(vessel.get(k, 0.0)) for k in FEATURE_ORDER],
            dtype=np.float32,
        )

    def predict(self, vessel: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predicts the behavioral segment for a single vessel.

        Returns:
            {
                "mmsi": str,
                "cluster_id": int,
                "segment": str (human-readable label),
                "segment_description": str,
                "confidence": float (distance to centroid, normalized),
            }
        """
        if self.model is None:
            # Fallback heuristic when K-Means model artifact is not loaded
            cid = _heuristic_segmentation(vessel)
            return {
                "mmsi": vessel.get("mmsi", "UNKNOWN"),
                "cluster_id": cid,
                "segment": SEGMENT_LABELS.get(cid, f"cluster_{cid}"),
                "segment_description": _get_segment_description(cid),
                "confidence": 0.85,
                "degraded": True,
            }

        X = self._build_feature_vector(vessel).reshape(1, -1)
        try:
            cluster_id = int(self.model.predict(X)[0])
            # Distance to all centroids (lower = more confident)
            distances = np.linalg.norm(self.model.cluster_centers_ - X, axis=1)
            sorted_dists = np.sort(distances)
            # Confidence: how much closer the best cluster is vs the 2nd best
            confidence = 1.0 - (sorted_dists[0] / (sorted_dists[1] + 1e-8))
            confidence = float(np.clip(confidence, 0.0, 1.0))

            return {
                "mmsi": vessel.get("mmsi", "UNKNOWN"),
                "cluster_id": cluster_id,
                "segment": SEGMENT_LABELS.get(cluster_id, f"cluster_{cluster_id}"),
                "segment_description": _get_segment_description(cluster_id),
                "confidence": round(confidence, 3),
            }
        except Exception as e:
            logger.error(f"K-Means segmentation failed: {e}")
            cid = _heuristic_segmentation(vessel)
            return {
                "mmsi": vessel.get("mmsi", "UNKNOWN"),
                "cluster_id": cid,
                "segment": SEGMENT_LABELS.get(cid, f"cluster_{cid}"),
                "segment_description": _get_segment_description(cid),
                "confidence": 0.5,
                "degraded": True,
            }

    def predict_batch(self, vessels: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Batch segmentation for a list of vessels.
        Uses numpy batch predict for efficiency.
        """
        if self.model is None:
            results = []
            for vessel in vessels:
                cid = _heuristic_segmentation(vessel)
                results.append({
                    "mmsi": vessel.get("mmsi", "UNKNOWN"),
                    "cluster_id": cid,
                    "segment": SEGMENT_LABELS.get(cid, f"cluster_{cid}"),
                    "segment_description": _get_segment_description(cid),
                    "degraded": True,
                })
            return results

        X = np.array([self._build_feature_vector(v) for v in vessels], dtype=np.float32)

        try:
            cluster_ids = self.model.predict(X)
        except Exception as e:
            logger.warning(f"K-Means batch prediction fallback: {e}")
            results = []
            for vessel in vessels:
                cid = _heuristic_segmentation(vessel)
                results.append({
                    "mmsi": vessel.get("mmsi", "UNKNOWN"),
                    "cluster_id": cid,
                    "segment": SEGMENT_LABELS.get(cid, f"cluster_{cid}"),
                    "segment_description": _get_segment_description(cid),
                    "degraded": True,
                })
            return results

        results = []
        for i, vessel in enumerate(vessels):
            cid = int(cluster_ids[i])
            results.append({
                "mmsi": vessel.get("mmsi", "UNKNOWN"),
                "cluster_id": cid,
                "segment": SEGMENT_LABELS.get(cid, f"cluster_{cid}"),
                "segment_description": _get_segment_description(cid),
            })
        return results

    def get_fleet_composition(self, vessels: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Returns the fleet composition breakdown for the command center dashboard.
        Example: {"deep_sea_trawler": 45%, "coastal_gillnetter": 30%, ...}
        """
        results = self.predict_batch(vessels)
        total = len(results)
        if total == 0:
            return {}

        counts: Dict[str, int] = {}
        for r in results:
            seg = r["segment"]
            counts[seg] = counts.get(seg, 0) + 1

        return {
            seg: {"count": cnt, "percentage": round(cnt / total * 100, 1)}
            for seg, cnt in counts.items()
        }


def _heuristic_segmentation(vessel: Dict[str, Any]) -> int:
    """Rules-based fallback archetype classifier when K-Means model is uninstantiated."""
    dist = float(vessel.get("max_distance_from_port_km", vessel.get("avg_daily_distance_km", 0.0)))
    duration = float(vessel.get("avg_trip_duration_hours", 0.0))
    fuel = float(vessel.get("avg_fuel_per_trip_liters", 0.0))

    if dist > 150 or duration > 72 or fuel > 600:
        return 1  # deep_sea_trawler
    elif dist > 80 or duration > 36:
        return 5  # offshore_longliner
    elif float(vessel.get("night_fishing_ratio", 0.0)) > 0.5:
        return 3  # opportunistic
    elif float(vessel.get("monsoon_activity_ratio", 0.0)) > 0.4:
        return 2  # seasonal_migrator
    elif duration < 12 and dist < 30:
        return 4  # artisanal_inshore
    return 0  # coastal_gillnetter


def _get_segment_description(cluster_id: int) -> str:
    descriptions = {
        0: "Coastal gillnetter: Short trips, near-shore, targets pelagic fish",
        1: "Deep-sea trawler: Long multi-day trips, high fuel burn, bottom trawling",
        2: "Seasonal migrator: Follows fish migration patterns across zones",
        3: "Opportunistic: Responds to real-time PFZ alerts and market prices",
        4: "Artisanal inshore: Day trips, traditional methods, near port",
        5: "Offshore longliner: Very long trips, targets tuna and large pelagics",
    }
    return descriptions.get(cluster_id, "Unknown vessel behavioral type")
