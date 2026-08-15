"""
BlueFish AI - Model 7 Service: Fuel-Efficient Route Optimization (A*)
======================================================================
Wraps the A* pathfinding model (model7.py) with a clean production interface.

The A* algorithm minimizes fuel cost by routing against ocean currents.
Grid nodes represent 0.25° × 0.25° cells of the Tamil Nadu EEZ.
Edge costs = base fuel cost + current resistance (opposing currents = higher cost).

Requires ocean current grids to be loaded via `load_environment()`.
Without grid data, falls back to a straight-line Haversine path.

Usage:
    service = RouteOptimizationService(model_registry.model7)
    result = service.optimize(9.5, 79.0, 12.0, 80.5)
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("bluefish.services.model7")

EARTH_RADIUS_KM = 6371.0


class RouteOptimizationService:
    """
    A* fuel-efficient route optimization service.
    Wraps RouteOptimizationModel from MODELS/model7/model7.py.
    """

    def __init__(self, model7_instance):
        self.model = model7_instance
        self._grid_loaded: bool = getattr(model7_instance, "grid_lat", None) is not None

    def optimize(
        self,
        start_lat: float,
        start_lon: float,
        dest_lat: float,
        dest_lon: float,
    ) -> Dict[str, Any]:
        """
        Computes the fuel-efficient route from start to destination.

        Returns:
            {
                "waypoints": List[{lat, lon, step}],
                "steps": int,
                "estimated_distance_km": float,
                "estimated_fuel_liters": float,
                "route_type": "optimized" | "straight_line",
                "degraded": bool,
            }
        """
        if not self._grid_loaded:
            return self._straight_line_fallback(start_lat, start_lon, dest_lat, dest_lon,
                                                 reason="ocean_grid_not_loaded")

        try:
            result = self.model.predict(start_lat, start_lon, dest_lat, dest_lon)
            route = result.get("route", [])
            steps = result.get("steps", len(route))

            dist_km = _haversine_km(start_lat, start_lon, dest_lat, dest_lon)
            # Estimate fuel: ~20 L/hr at 15 km/h → ~1.33 L/km baseline + 20% routing overhead
            fuel_liters = round(dist_km * 1.33 * 1.20, 1)

            return {
                "waypoints": route,
                "steps": steps,
                "estimated_distance_km": round(dist_km, 2),
                "estimated_fuel_liters": fuel_liters,
                "route_type": "optimized",
                "degraded": False,
            }
        except Exception as e:
            logger.warning(f"A* route optimization failed: {e}. Falling back to straight-line.")
            return self._straight_line_fallback(start_lat, start_lon, dest_lat, dest_lon,
                                                 reason=str(e))

    def _straight_line_fallback(
        self,
        start_lat: float, start_lon: float,
        dest_lat: float, dest_lon: float,
        reason: str = "model_not_available",
    ) -> Dict[str, Any]:
        """Returns a 2-waypoint straight-line route when A* is unavailable."""
        dist_km = _haversine_km(start_lat, start_lon, dest_lat, dest_lon)
        fuel_liters = round(dist_km * 1.33, 1)

        return {
            "waypoints": [
                {"lat": start_lat, "lon": start_lon, "step": 0},
                {"lat": dest_lat, "lon": dest_lon, "step": 1},
            ],
            "steps": 2,
            "estimated_distance_km": round(dist_km, 2),
            "estimated_fuel_liters": fuel_liters,
            "route_type": "straight_line",
            "degraded": True,
            "degraded_reason": reason,
        }

    def to_geojson(self, route_result: Dict[str, Any]) -> Dict[str, Any]:
        """Converts route waypoints to a GeoJSON LineString for the map."""
        waypoints = route_result.get("waypoints", [])
        coordinates = [[wp["lon"], wp["lat"]] for wp in waypoints]
        return {
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": coordinates,
            },
            "properties": {
                "route_type": route_result.get("route_type", "unknown"),
                "steps": route_result.get("steps", 0),
                "estimated_distance_km": route_result.get("estimated_distance_km", 0.0),
                "estimated_fuel_liters": route_result.get("estimated_fuel_liters", 0.0),
                "degraded": route_result.get("degraded", False),
            },
        }


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Computes great-circle distance between two lat/lon points in kilometers."""
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(a))
