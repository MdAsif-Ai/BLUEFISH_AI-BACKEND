"""
BlueFish AI - Model 10 Service: Collision Detection (CPA/TCPA Kinematics)
==========================================================================
Wraps the CPA/TCPA collision risk model (model10.py).

CPA (Closest Point of Approach): Minimum distance two vessels will reach
given their current positions, speeds, and headings.
TCPA (Time to CPA): How many minutes until they reach the CPA.

A collision risk is flagged when:
  - CPA < COLLISION_CPA_THRESHOLD_KM (default: 0.5 km)
  - TCPA < COLLISION_TCPA_THRESHOLD_MIN (default: 15 minutes)
  - TCPA > 0 (vessels are still converging, not diverging)

This is PURE KINEMATIC MATH — no ML inference needed.
Always available. O(N²) complexity — optimized with spatial pre-filtering.

Usage:
    service = CollisionDetectionService(model_registry.model10)
    alerts = service.detect_all_pairs(vessels_list)
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("bluefish.services.model10")

DEFAULT_CPA_THRESHOLD_KM = 0.5
DEFAULT_TCPA_THRESHOLD_MIN = 15.0


class CollisionDetectionService:
    """
    CPA/TCPA collision detection service.
    Wraps CollisionDetectionModel from MODELS/model10/model10.py.
    Also exposes a direct fallback implementation if model10 fails.
    """

    def __init__(
        self,
        model10_instance,
        cpa_threshold_km: float = DEFAULT_CPA_THRESHOLD_KM,
        tcpa_threshold_min: float = DEFAULT_TCPA_THRESHOLD_MIN,
    ):
        self.model = model10_instance
        self.cpa_threshold = cpa_threshold_km
        self.tcpa_threshold = tcpa_threshold_min

    def detect_all_pairs(
        self,
        vessels: List[Dict[str, Any]],
        max_spatial_filter_km: float = 50.0,
    ) -> List[Dict[str, Any]]:
        """
        Detects collision risks across all vessel pairs.

        Optimization: First filters pairs by proximity (haversine < 50km)
        to avoid running full CPA/TCPA on the full O(N²) space.

        Args:
            vessels: List of vessel dicts with keys:
                     mmsi, lat, lon, speed (knots), heading (degrees)
            max_spatial_filter_km: Only evaluate pairs within this distance

        Returns:
            List of collision alert dicts sorted by TCPA (most urgent first).
        """
        if len(vessels) < 2:
            return []

        collision_alerts = []

        try:
            if self.model is not None:
                # Delegate to model10's built-in pairwise detection
                result = self.model.detect_collisions(
                    vessels,
                    cpa_threshold_km=self.cpa_threshold,
                    tcpa_threshold_min=self.tcpa_threshold,
                )
                collision_alerts = result if isinstance(result, list) else []
            else:
                # Direct fallback implementation
                collision_alerts = self._detect_direct(vessels, max_spatial_filter_km)
        except Exception as e:
            logger.warning(f"model10.detect_collisions failed: {e}. Using direct fallback.")
            collision_alerts = self._detect_direct(vessels, max_spatial_filter_km)

        # Sort by TCPA ascending (most urgent first)
        collision_alerts.sort(key=lambda x: x.get("tcpa_minutes", float("inf")))
        return collision_alerts

    def _detect_direct(
        self,
        vessels: List[Dict[str, Any]],
        max_spatial_filter_km: float,
    ) -> List[Dict[str, Any]]:
        """
        Direct CPA/TCPA implementation — runs when model10 is unavailable.
        Spatial pre-filtering: skip pairs > 50km apart.
        """
        alerts = []
        n = len(vessels)

        for i in range(n):
            for j in range(i + 1, n):
                v1 = vessels[i]
                v2 = vessels[j]

                # Spatial pre-filter: haversine distance
                d_km = _haversine_km(
                    v1.get("lat", 0), v1.get("lon", 0),
                    v2.get("lat", 0), v2.get("lon", 0),
                )
                if d_km > max_spatial_filter_km:
                    continue

                cpa, tcpa = _compute_cpa_tcpa(v1, v2)
                if tcpa is None or tcpa <= 0 or cpa is None:
                    continue

                if cpa < self.cpa_threshold and tcpa < self.tcpa_threshold:
                    severity = "HIGH" if (cpa < 0.2 and tcpa < 5) else "MEDIUM"
                    mid_lat = (v1.get("lat", 0) + v2.get("lat", 0)) / 2
                    mid_lon = (v1.get("lon", 0) + v2.get("lon", 0)) / 2
                    alerts.append({
                        "vessel_1_mmsi": v1.get("mmsi", "UNKNOWN"),
                        "vessel_2_mmsi": v2.get("mmsi", "UNKNOWN"),
                        "cpa_km": round(cpa, 3),
                        "tcpa_minutes": round(tcpa, 1),
                        "collision_lat": round(mid_lat, 5),
                        "collision_lon": round(mid_lon, 5),
                        "severity": severity,
                        "current_distance_km": round(d_km, 2),
                    })

        return alerts

    def to_redis_alerts(
        self,
        collision_alerts: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Formats collision alerts for Redis Pub/Sub publication.
        Each alert gets a channel name and timestamp added.
        """
        from datetime import datetime, timezone

        now = datetime.now(timezone.utc).isoformat()
        formatted = []
        for alert in collision_alerts:
            formatted.append({
                **alert,
                "alert_type": "collision",
                "detected_at": now,
                "redis_channel": "bluefish:safety_alerts",
            })
        return formatted


# ── CPA/TCPA math (direct implementation) ────────────────────────────────────

def _compute_cpa_tcpa(
    v1: Dict[str, Any],
    v2: Dict[str, Any],
) -> Tuple[Optional[float], Optional[float]]:
    """
    Computes CPA (km) and TCPA (minutes) for two vessels using
    relative velocity kinematic equations.

    Coordinate system: flat-Earth approximation valid for distances <100km.
    Speed input: knots → converted to km/min internally.
    """
    KNOTS_TO_KM_PER_MIN = 1.852 / 60.0

    # Positions in km (using flat-Earth approximation around the mean position)
    lat1, lon1 = v1.get("lat", 0.0), v1.get("lon", 0.0)
    lat2, lon2 = v2.get("lat", 0.0), v2.get("lon", 0.0)
    lat_mid = (lat1 + lat2) / 2.0
    cos_lat = math.cos(math.radians(lat_mid))
    KM_PER_DEG_LAT = 111.0
    KM_PER_DEG_LON = 111.0 * cos_lat

    x1, y1 = lon1 * KM_PER_DEG_LON, lat1 * KM_PER_DEG_LAT
    x2, y2 = lon2 * KM_PER_DEG_LON, lat2 * KM_PER_DEG_LAT

    # Velocity components (km/min)
    def vel_components(speed_knots: float, heading_deg: float) -> Tuple[float, float]:
        spd = float(speed_knots) * KNOTS_TO_KM_PER_MIN
        hdg = math.radians(float(heading_deg))
        return spd * math.sin(hdg), spd * math.cos(hdg)  # (vx, vy)

    vx1, vy1 = vel_components(v1.get("speed", 0), v1.get("heading", 0))
    vx2, vy2 = vel_components(v2.get("speed", 0), v2.get("heading", 0))

    # Relative position and velocity
    dx = x1 - x2
    dy = y1 - y2
    dvx = vx1 - vx2
    dvy = vy1 - vy2

    dv_sq = dvx**2 + dvy**2
    if dv_sq < 1e-10:
        # Vessels moving at the same velocity — CPA is current distance
        cpa = math.sqrt(dx**2 + dy**2)
        return cpa, float("inf")

    # TCPA: time to CPA
    tcpa = -(dx * dvx + dy * dvy) / dv_sq

    if tcpa < 0:
        return None, None  # Vessels are diverging

    # CPA: distance at TCPA
    cpa_x = dx + tcpa * dvx
    cpa_y = dy + tcpa * dvy
    cpa = math.sqrt(cpa_x**2 + cpa_y**2)

    return cpa, tcpa


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
    return 2 * 6371.0 * math.asin(math.sqrt(a))
