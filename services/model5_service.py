"""
BlueFish AI - Model 5 Service: Fleet Density (DBSCAN Clustering)
=================================================================
Wraps DBSCAN spatial clustering for fleet overcrowding detection.

Model 5 takes a list of live vessel GPS positions (or DataFrame)
and identifies clusters of overcrowded vessels. Each cluster is labeled with
a severity level based on vessel count and spatial density.

Usage:
    service = FleetDensityService(model_registry.model5)
    result = service.predict(vessels_list, target_date)
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

logger = logging.getLogger("bluefish.services.model5")

SEVERITY_THRESHOLDS = {
    "SEVERE": 50,    # ≥50 vessels in a cluster
    "HIGH": 25,      # ≥25 vessels
    "MODERATE": 10,  # ≥10 vessels
    "LOW": 3,        # ≥3 vessels
}


class FleetDensityModel:
    """
    BlueFish AI - Model 5: Fleet Density & Overcrowding Detector
    Wraps DBSCAN spatial clustering with Haversine metric.
    """

    def __init__(
        self,
        eps_km: float = 5.0,
        min_vessels: int = 10,
        lat_min: float = 6.0,
        lat_max: float = 23.0,
        lon_min: float = 68.0,
        lon_max: float = 89.0,
    ):
        self.eps_km = eps_km
        self.min_vessels = min_vessels
        self.lat_min = lat_min
        self.lat_max = lat_max
        self.lon_min = lon_min
        self.lon_max = lon_max
        self.earth_radius_km = 6371.0088
        self._model = None

    def _to_dataframe(self, data: Union[pd.DataFrame, List[Dict[str, Any]]]) -> pd.DataFrame:
        if isinstance(data, pd.DataFrame):
            df = data.copy()
        else:
            df = pd.DataFrame(data)

        if df.empty:
            return df

        # Normalize lat/lon column names
        rename_map = {}
        for col in df.columns:
            col_lower = str(col).lower()
            if col_lower in ("lat", "latitude", "cell_ll_lat"):
                rename_map[col] = "cell_ll_lat"
            elif col_lower in ("lon", "lng", "longitude", "cell_ll_lon"):
                rename_map[col] = "cell_ll_lon"
            elif col_lower in ("fishing_hours", "hours"):
                rename_map[col] = "fishing_hours"
        df = df.rename(columns=rename_map)

        if "fishing_hours" not in df.columns:
            df["fishing_hours"] = 1.0

        return df

    def _cluster(self, day_df: pd.DataFrame) -> pd.DataFrame:
        if day_df.empty or len(day_df) < self.min_vessels:
            day_df["cluster_id"] = -1
            return day_df

        coords_rad = np.radians(day_df[["cell_ll_lat", "cell_ll_lon"]].to_numpy())
        eps_rad = self.eps_km / self.earth_radius_km

        self._model = DBSCAN(
            eps=eps_rad,
            min_samples=self.min_vessels,
            metric="haversine",
            algorithm="ball_tree",
        )

        day_df["cluster_id"] = self._model.fit_predict(coords_rad)
        return day_df

    def _format_output(self, day_df: pd.DataFrame) -> Dict[str, Any]:
        if day_df.empty:
            return {"zones": [], "boats": [], "unclustered_count": 0}

        clusters = day_df[day_df["cluster_id"] != -1].copy()
        zones = []

        if "cluster_id" in day_df.columns:
            for cid in clusters["cluster_id"].unique():
                c = clusters[clusters["cluster_id"] == cid]
                size = len(c)

                center_lat = float(c["cell_ll_lat"].mean())
                center_lon = float(c["cell_ll_lon"].mean())

                lat_km = (c["cell_ll_lat"].max() - c["cell_ll_lat"].min()) * 111.0
                lon_km = (c["cell_ll_lon"].max() - c["cell_ll_lon"].min()) * 111.0 * np.cos(np.radians(center_lat))
                area = max(float(lat_km * lon_km), 0.1)
                density = size / area

                if density > 5:
                    severity = "SEVERE"
                elif density > 2:
                    severity = "HIGH"
                else:
                    severity = "MODERATE"

                zones.append({
                    "cluster_id": int(cid),
                    "vessels": int(size),
                    "vessel_count": int(size),
                    "center_lat": round(center_lat, 4),
                    "center_lon": round(center_lon, 4),
                    "area_km2": round(area, 2),
                    "radius_km": round(np.sqrt(area / np.pi), 2),
                    "density": round(density, 2),
                    "severity": severity,
                })

        zones.sort(key=lambda x: x["density"], reverse=True)

        boats = []
        for _, row in day_df.iterrows():
            boat_dict = {
                "lat": float(row["cell_ll_lat"]),
                "lon": float(row["cell_ll_lon"]),
                "cluster_id": int(row.get("cluster_id", -1)),
                "mmsi": row.get("mmsi", "UNKNOWN"),
                "speed": float(row.get("speed", 0.0)),
                "heading": float(row.get("heading", 0.0)),
            }
            boats.append(boat_dict)

        unclustered = sum(1 for b in boats if b["cluster_id"] == -1)

        return {
            "zones": zones,
            "boats": boats,
            "unclustered_count": unclustered,
        }

    def predict(self, data: Union[pd.DataFrame, List[Dict[str, Any]]], target_date: Optional[str] = None) -> Dict[str, Any]:
        """
        Main API Method. Pass raw dataframe or list of vessel dicts, get JSON output.
        """
        df = self._to_dataframe(data)
        clustered_df = self._cluster(df)
        return self._format_output(clustered_df)


class FleetDensityService:
    """
    DBSCAN-based fleet density clustering service.
    Wraps FleetDensityModel.
    """

    def __init__(self, model5_instance: Optional[Any] = None):
        if model5_instance is None:
            self.model = FleetDensityModel()
        else:
            self.model = model5_instance

    def predict(
        self,
        vessels: Union[List[Dict[str, Any]], pd.DataFrame],
        target_date: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Clusters live vessel positions to identify overcrowded zones.
        """
        if isinstance(vessels, list) and not vessels:
            return {"zones": [], "boats": [], "unclustered_count": 0, "total_vessels": 0, "overcrowded_zones": 0}

        try:
            date_str = target_date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
            result = self.model.predict(vessels, date_str)

            zones = result.get("zones", [])
            for zone in zones:
                count = zone.get("vessel_count", zone.get("vessels", 0))
                if "severity" not in zone or not zone["severity"]:
                    zone["severity"] = _classify_severity(count)

            total_vessels = len(vessels) if isinstance(vessels, list) else len(vessels)

            return {
                "zones": zones,
                "boats": result.get("boats", []),
                "unclustered_count": result.get("unclustered_count", 0),
                "total_vessels": total_vessels,
                "overcrowded_zones": sum(1 for z in zones if z.get("severity") in ("HIGH", "SEVERE")),
            }
        except Exception as e:
            logger.error(f"DBSCAN density prediction failed: {e}", exc_info=True)
            return {"zones": [], "boats": [], "unclustered_count": 0, "error": str(e)}

    def to_geojson(self, density_result: Dict[str, Any]) -> Dict[str, Any]:
        features = []

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
                    "vessel_count": zone.get("vessel_count", zone.get("vessels", 0)),
                    "severity": zone.get("severity", "LOW"),
                    "radius_km": zone.get("radius_km", 5.0),
                    "cluster_id": zone.get("cluster_id", -1),
                },
            })

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
