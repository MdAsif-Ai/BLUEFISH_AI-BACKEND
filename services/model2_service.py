"""
BlueFish AI - Model 2 Service: Ocean Fronts & Eddies Detection
===============================================================
Wraps the pure-math ocean front and eddy detection module (model2.py).

Model 2 uses:
  - Sobel gradient filtering on SST fields → detects thermal fronts
  - Okubo-Weiss parameter (Q) on velocity fields → detects eddies

Input: xarray Dataset with variables: sst, uo (u-velocity), vo (v-velocity)
Output: GeoJSON FeatureCollection with detected fronts (LineStrings) and
        eddies (Points/Polygons) for the CesiumJS map.

Usage:
    service = OceanFeatureService(model_registry.model2)
    result = service.detect(netcdf_dataset, date_str)
    geojson = service.to_geojson(result)
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger("bluefish.services.model2")


class OceanFeatureService:
    """
    Wraps the OceanFrontEddyModel (model2.py) for production use.
    Normalizes inputs to xarray Dataset format expected by model2.
    """

    def __init__(self, model2_instance):
        """
        Args:
            model2_instance: An instantiated OceanFrontEddyModel() object
                             from MODELS/model2/model2.py
        """
        self.model = model2_instance

    def detect_from_dataset(self, ds, date_str: Optional[str] = None) -> Dict[str, Any]:
        """
        Runs front and eddy detection on an xarray Dataset.

        Args:
            ds: xarray.Dataset with variables: sst, uo, vo
                and coordinates: latitude, longitude
            date_str: ISO date string for labeling (YYYY-MM-DD)

        Returns:
            Dict with 'fronts' and 'eddies' arrays.
        """
        if ds is None:
            return {"fronts": [], "eddies": [], "error": "No dataset provided"}

        try:
            result = self.model.predict(ds)
            return {
                "date": date_str or datetime.utcnow().strftime("%Y-%m-%d"),
                "fronts": result.get("fronts", []),
                "eddies": result.get("eddies", []),
                "front_count": len(result.get("fronts", [])),
                "eddy_count": len(result.get("eddies", [])),
            }
        except Exception as e:
            logger.error(f"Ocean feature detection failed: {e}", exc_info=True)
            return {"fronts": [], "eddies": [], "error": str(e)}

    def detect_from_grids(
        self,
        sst_grid: np.ndarray,
        u_grid: np.ndarray,
        v_grid: np.ndarray,
        lat_coords: np.ndarray,
        lon_coords: np.ndarray,
        date_str: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Convenience method when data is available as numpy arrays rather than
        an xarray Dataset. Builds the Dataset internally.
        """
        try:
            import xarray as xr

            ds = xr.Dataset(
                {
                    "sst": (["latitude", "longitude"], sst_grid),
                    "uo": (["latitude", "longitude"], u_grid),
                    "vo": (["latitude", "longitude"], v_grid),
                },
                coords={"latitude": lat_coords, "longitude": lon_coords},
            )
            return self.detect_from_dataset(ds, date_str)
        except Exception as e:
            logger.error(f"Grid-to-dataset conversion failed: {e}")
            return {"fronts": [], "eddies": [], "error": str(e)}

    def to_geojson(self, detection_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Converts the detection result to a GeoJSON FeatureCollection for Redis caching.

        Fronts → GeoJSON LineStrings
        Eddies → GeoJSON Points with rotation direction property
        """
        features = []

        # Fronts as LineStrings
        for front in detection_result.get("fronts", []):
            coords = front.get("coordinates", [])
            if not coords or len(coords) < 2:
                continue
            features.append({
                "type": "Feature",
                "geometry": {
                    "type": "LineString",
                    "coordinates": [[c["lon"], c["lat"]] for c in coords],
                },
                "properties": {
                    "type": "front",
                    "gradient_magnitude": front.get("gradient_magnitude", 0.0),
                    "sst_difference": front.get("sst_difference", 0.0),
                },
            })

        # Eddies as Points
        for eddy in detection_result.get("eddies", []):
            features.append({
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [eddy.get("lon", 0.0), eddy.get("lat", 0.0)],
                },
                "properties": {
                    "type": "eddy",
                    "rotation": eddy.get("rotation", "cyclonic"),
                    "radius_km": eddy.get("radius_km", 50.0),
                    "okubo_weiss": eddy.get("okubo_weiss", 0.0),
                },
            })

        return {
            "type": "FeatureCollection",
            "features": features,
            "date": detection_result.get("date"),
            "meta": {
                "front_count": detection_result.get("front_count", 0),
                "eddy_count": detection_result.get("eddy_count", 0),
            },
        }
