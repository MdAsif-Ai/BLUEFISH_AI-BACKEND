"""
BlueFish AI - Model 5 Service: Fleet Density (DBSCAN Clustering)
=================================================================
Wraps model5.py DBSCAN spatial clustering for fleet overcrowding detection.

Model 5 takes a list of live vessel GPS positions (from the Redis Geo-index)
and identifies clusters of overcrowded vessels. Each cluster is labeled with
a severity level based on vessel count and spatial density.

Usage:
    service = FleetDensityService(model_registry.model5)
    result = service.predict(vessels_list, target_date)
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger("bluefish.services.model5")

# Overcrowding severity thresholds
SEVERITY_THRESHOLDS = {
    "SEVERE": 50,    # ≥50 vessels in a cluster
    "HIGH": 25,      # ≥25 vessels
    "MODERATE": 10,  # ≥10 vessels
    "LOW": 3,        # ≥3 vessels
}


class FleetDensityService:
    """
    DBSCAN-based fleet density clustering service.
    Wraps FleetDensityModel from MODELS/model5/model5.py.
    """

    def __init__(self, model5_instance):
        self.model = model5_instance

    def predict(
        self,
        vessels: List[Dict[str, Any]],
        target_date: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Clusters live vessel positions to identify overcrowded zones.

        Args:
            vessels: List of dicts with keys: mmsi, lat, lon, speed, heading, timestamp
            target_date: Date string for context labeling

        Returns:
            {
                "zones": List of cluster summary dicts with severity labels,
                "boats": List of individual boat dicts with cluster_id,
                "unclustered_count": int  (vessels not in any dense cluster),
            }
        """
        if not vessels:
            return {"zones": [], "boats": [], "unclustered_count": 0}

        try:
            result = self.model.predict(vessels, target_date or datetime.now(timezone.utc).strftime("%Y-%m-%d"))
            # Enrich zones with severity labels
            zones = result.get("zones", [])
            for zone in zones:
                count = zone.get("vessel_count", 0)
                zone["severity"] = _classify_severity(count)

            return {
                "zones": zones,
                "boats": result.get("boats", []),
                "unclustered_count": result.get("unclustered_count", 0),
                "total_vessels": len(vessels),
                "overcrowded_zones": sum(1 for z in zones if z.get("severity") in ("HIGH", "SEVERE")),
            }
        except Exception as e:
            logger.error(f"DBSCAN density prediction failed: {e}", exc_info=True)
            return {"zones": [], "boats": [], "unclustered_count": len(vessels), "error": str(e)}

    def to_geojson(self, density_result: Dict[str, Any]) -> Dict[str, Any]:
        """Converts density result to GeoJSON for the CesiumJS map heatmap layer."""
        features = []

        # Zones as Polygons or Points (depending on what model5 returns)
        for zone in density_result.get("zones", []):
            center_lat = zone.get("center_lat", 0.0)
            center_lon = zone.get("center_lon", 0.0)
            features.append({
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [center_lon, center_lat],
                },
                "properties": {
                    "type": "density_zone",
                    "vessel_count": zone.get("vessel_count", 0),
                    "severity": zone.get("severity", "LOW"),
                    "radius_km": zone.get("radius_km", 5.0),
                    "cluster_id": zone.get("cluster_id", -1),
                },
            })

        # Individual boats
        for boat in density_result.get("boats", []):
            features.append({
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [boat.get("lon", 0.0), boat.get("lat", 0.0)],
                },
                "properties": {
                    "type": "vessel",
                    "mmsi": boat.get("mmsi"),
                    "cluster_id": boat.get("cluster_id", -1),
                    "speed": boat.get("speed", 0.0),
                    "heading": boat.get("heading", 0.0),
                },
            })

        return {
            "type": "FeatureCollection",
            "features": features,
            "meta": {
                "total_vessels": density_result.get("total_vessels", 0),
                "overcrowded_zones": density_result.get("overcrowded_zones", 0),
            },
        }


def _classify_severity(vessel_count: int) -> str:
    if vessel_count >= SEVERITY_THRESHOLDS["SEVERE"]:
        return "SEVERE"
    elif vessel_count >= SEVERITY_THRESHOLDS["HIGH"]:
        return "HIGH"
    elif vessel_count >= SEVERITY_THRESHOLDS["MODERATE"]:
        return "MODERATE"
    elif vessel_count >= SEVERITY_THRESHOLDS["LOW"]:
        return "LOW"
    return "MINIMAL"
