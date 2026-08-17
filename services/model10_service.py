"""
BlueFish AI - Model 10 Service: Collision Detection (CPA/TCPA Kinematics)
==========================================================================
Wraps the CPA/TCPA collision risk model.

CPA (Closest Point of Approach): Minimum distance two vessels will reach
given their current positions, speeds, and headings.
TCPA (Time to CPA): How many minutes until they reach the CPA.

A collision risk is flagged when:
  - CPA < COLLISION_CPA_THRESHOLD_KM (default: 0.5 km)
  - TCPA < COLLISION_TCPA_THRESHOLD_MIN (default: 15 minutes)
  - TCPA > 0 (vessels are converging)

Uses Spatial Hashing + CPA/TCPA kinematics for O(N) scalability.
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("bluefish.services.model10")

DEFAULT_CPA_THRESHOLD_KM = 0.5
DEFAULT_TCPA_THRESHOLD_MIN = 15.0


class CollisionDetectionModel:
    """
    BlueFish AI - Model 10: Vessel Collision & Near-Miss Detector
    Uses spatial hashing + CPA/TCPA kinematics for O(N) scalability.
    """

    def __init__(
        self,
        cpa_threshold_km: float = DEFAULT_CPA_THRESHOLD_KM,
        tcpa_threshold_min: float = DEFAULT_TCPA_THRESHOLD_MIN,
        grid_size_km: float = 20.0,
    ):
        self.cpa_threshold_km = cpa_threshold_km
        self.tcpa_threshold_min = tcpa_threshold_min
        self.grid_size_km = grid_size_km
        self.earth_radius_km = 6371.0

    def _to_xy_km(self, lat: float, lon: float, ref_lat: float, ref_lon: float) -> Tuple[float, float]:
        """Converts Lat/Lon to flat X/Y in kilometers relative to a reference point."""
        x = lon * (math.pi / 180.0) * self.earth_radius_km * math.cos(math.radians(ref_lat))
        y = lat * (math.pi / 180.0) * self.earth_radius_km
        ref_x = ref_lon * (math.pi / 180.0) * self.earth_radius_km * math.cos(math.radians(ref_lat))
        ref_y = ref_lat * (math.pi / 180.0) * self.earth_radius_km
        return x - ref_x, y - ref_y

    def _calculate_cpa_tcpa(self, v1: Dict[str, Any], v2: Dict[str, Any]) -> Tuple[float, float]:
        """Calculates Closest Point of Approach (CPA) and Time to CPA (TCPA)."""
        ref_lat, ref_lon = v1["lat"], v1["lon"]
        x2, y2 = self._to_xy_km(v2["lat"], v2["lon"], ref_lat, ref_lon)

        spd1_kmm = v1.get("speed", 0.0) * 1.852 / 60.0
        spd2_kmm = v2.get("speed", 0.0) * 1.852 / 60.0

        vx1 = spd1_kmm * math.sin(math.radians(v1.get("heading", 0.0)))
        vy1 = spd1_kmm * math.cos(math.radians(v1.get("heading", 0.0)))
        vx2 = spd2_kmm * math.sin(math.radians(v2.get("heading", 0.0)))
        vy2 = spd2_kmm * math.cos(math.radians(v2.get("heading", 0.0)))

        dx = x2
        dy = y2
        dvx = vx2 - vx1
        dvy = vy2 - vy1

        rel_speed_sq = dvx**2 + dvy**2
        if rel_speed_sq < 1e-6:
            tcpa = 0.0
        else:
            tcpa = -(dx * dvx + dy * dvy) / rel_speed_sq

        if tcpa <= 0:
            cpa_dist = math.sqrt(dx**2 + dy**2)
            tcpa = 0.0
        else:
            cpa_x = dx + dvx * tcpa
            cpa_y = dy + dvy * tcpa
            cpa_dist = math.sqrt(cpa_x**2 + cpa_y**2)

        return cpa_dist, tcpa

    def predict(self, vessels: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Main API Method. Pass a list of active vessels, get collision alerts.
        Required keys: 'mmsi', 'lat', 'lon', 'speed', 'heading'
        """
        if len(vessels) < 2:
            return {"alerts": [], "vessels_checked": len(vessels)}

        alerts = []
        grid = defaultdict(list)

        for v in vessels:
            grid_lat = int(v["lat"] * (111.0 / self.grid_size_km))
            grid_lon = int(v["lon"] * (111.0 / self.grid_size_km))
            grid[(grid_lat, grid_lon)].append(v)

        checked_pairs = set()

        for (glat, glon), cell_vessels in grid.items():
            nearby_vessels = []
            for i in [-1, 0, 1]:
                for j in [-1, 0, 1]:
                    nearby_vessels.extend(grid.get((glat + i, glon + j), []))

            for v1 in cell_vessels:
                for v2 in nearby_vessels:
                    mmsi1 = v1.get("mmsi")
                    mmsi2 = v2.get("mmsi")
                    if mmsi1 == mmsi2 or mmsi1 is None or mmsi2 is None:
                        continue

                    pair_id = tuple(sorted((mmsi1, mmsi2)))
                    if pair_id in checked_pairs:
                        continue
                    checked_pairs.add(pair_id)

                    cpa, tcpa = self._calculate_cpa_tcpa(v1, v2)

                    if cpa <= self.cpa_threshold_km and tcpa <= self.tcpa_threshold_min and tcpa >= 0:
                        collision_lat = v1["lat"] + (v1.get("speed", 0) * 1.852 / 60.0 * math.cos(math.radians(v1.get("heading", 0))) * tcpa) / 111.0
                        collision_lon = v1["lon"] + (v1.get("speed", 0) * 1.852 / 60.0 * math.sin(math.radians(v1.get("heading", 0))) * tcpa) / (111.0 * math.cos(math.radians(v1["lat"])))

                        alerts.append({
                            "vessel_1_mmsi": mmsi1,
                            "vessel_2_mmsi": mmsi2,
                            "cpa_km": round(cpa, 3),
                            "tcpa_min": round(tcpa, 1),
                            "tcpa_minutes": round(tcpa, 1),
                            "collision_lat": round(collision_lat, 4),
                            "collision_lon": round(collision_lon, 4),
                            "severity": "HIGH" if cpa < 0.2 else "MEDIUM",
                        })

        return {"alerts": alerts, "vessels_checked": len(vessels)}

    def detect_collisions(
        self,
        vessels: List[Dict[str, Any]],
        cpa_threshold_km: Optional[float] = None,
        tcpa_threshold_min: Optional[float] = None,
    ) -> List[Dict[str, Any]]:

        orig_cpa = self.cpa_threshold_km
        orig_tcpa = self.tcpa_threshold_min
        if cpa_threshold_km is not None:
            self.cpa_threshold_km = cpa_threshold_km
        if tcpa_threshold_min is not None:
            self.tcpa_threshold_min = tcpa_threshold_min

        res = self.predict(vessels)

        self.cpa_threshold_km = orig_cpa
        self.tcpa_threshold_min = orig_tcpa

        return res.get("alerts", [])


class CollisionDetectionService:
    """
    CPA/TCPA collision detection service.
    Wraps CollisionDetectionModel.
    """

    def __init__(
        self,
        model10_instance: Optional[Any] = None,
        cpa_threshold_km: float = DEFAULT_CPA_THRESHOLD_KM,
        tcpa_threshold_min: float = DEFAULT_TCPA_THRESHOLD_MIN,
    ):
        if model10_instance is None:
            self.model = CollisionDetectionModel(
                cpa_threshold_km=cpa_threshold_km,
                tcpa_threshold_min=tcpa_threshold_min,
            )
        else:
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
        """
        if len(vessels) < 2:
            return []

        try:
            if hasattr(self.model, "detect_collisions"):
                collision_alerts = self.model.detect_collisions(
                    vessels,
                    cpa_threshold_km=self.cpa_threshold,
                    tcpa_threshold_min=self.tcpa_threshold,
                )
            elif hasattr(self.model, "predict"):
                res = self.model.predict(vessels)
                collision_alerts = res.get("alerts", [])
            else:
                collision_alerts = self._detect_direct(vessels, max_spatial_filter_km)
        except Exception as e:
            logger.warning(f"model10 detection failed: {e}. Using direct fallback.")
            collision_alerts = self._detect_direct(vessels, max_spatial_filter_km)

        if not isinstance(collision_alerts, list):
            collision_alerts = []

        collision_alerts.sort(key=lambda x: x.get("tcpa_minutes", x.get("tcpa_min", float("inf"))))
        return collision_alerts

    def _detect_direct(
        self,
        vessels: List[Dict[str, Any]],
        max_spatial_filter_km: float,
    ) -> List[Dict[str, Any]]:
        alerts = []
        n = len(vessels)

        for i in range(n):
            for j in range(i + 1, n):
                v1 = vessels[i]
                v2 = vessels[j]

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
                        "tcpa_min": round(tcpa, 1),
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


def _compute_cpa_tcpa(
    v1: Dict[str, Any],
    v2: Dict[str, Any],
) -> Tuple[Optional[float], Optional[float]]:
    KNOTS_TO_KM_PER_MIN = 1.852 / 60.0

    lat1, lon1 = v1.get("lat", 0.0), v1.get("lon", 0.0)
    lat2, lon2 = v2.get("lat", 0.0), v2.get("lon", 0.0)
    lat_mid = (lat1 + lat2) / 2.0
    cos_lat = math.cos(math.radians(lat_mid))
    KM_PER_DEG_LAT = 111.0
    KM_PER_DEG_LON = 111.0 * cos_lat

    x1, y1 = lon1 * KM_PER_DEG_LON, lat1 * KM_PER_DEG_LAT
    x2, y2 = lon2 * KM_PER_DEG_LON, lat2 * KM_PER_DEG_LAT

    def vel_components(speed_knots: float, heading_deg: float) -> Tuple[float, float]:
        spd = float(speed_knots) * KNOTS_TO_KM_PER_MIN
        hdg = math.radians(float(heading_deg))
        return spd * math.sin(hdg), spd * math.cos(hdg)

    vx1, vy1 = vel_components(v1.get("speed", 0), v1.get("heading", 0))
    vx2, vy2 = vel_components(v2.get("speed", 0), v2.get("heading", 0))

    dx = x1 - x2
    dy = y1 - y2
    dvx = vx1 - vx2
    dvy = vy1 - vy2

    dv_sq = dvx**2 + dvy**2
    if dv_sq < 1e-10:
        cpa = math.sqrt(dx**2 + dy**2)
        return cpa, float("inf")

    tcpa = -(dx * dvx + dy * dvy) / dv_sq

    if tcpa < 0:
        return None, None

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
