"""
BlueFish AI - Model 7 Service: Fuel-Efficient Route Optimization (A*)
======================================================================
Wraps the A* pathfinding model with a clean production interface.

The A* algorithm minimizes fuel cost by routing against ocean currents.
Grid nodes represent cells of the EEZ region.
Edge costs = base fuel cost + current resistance (opposing currents = higher cost).
"""

from __future__ import annotations

import heapq
import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("bluefish.services.model7")

EARTH_RADIUS_KM = 6371.0


class RouteOptimizationModel:
    """
    BlueFish AI - Model 7: Fuel-Efficient Route Optimization
    Uses A* Pathfinding weighted by ocean current resistance and bathymetry.
    """

    def __init__(
        self,
        cost_per_km: float = 1.0,
        cost_against_current: float = 10.0,
        cost_with_current: float = 0.1,
        min_depth_m: float = -5.0,
    ):
        self.cost_per_km = cost_per_km
        self.cost_against_current = cost_against_current
        self.cost_with_current = cost_with_current
        self.min_depth_m = min_depth_m
        self.grid_lat = None
        self.grid_lon = None
        self.uo = None
        self.vo = None
        self.depth = None

    def load_environment(self, currents_ds: Any, bathymetry_ds: Optional[Any] = None):
        """Loads daily currents and static bathymetry into memory."""
        uo_da = currents_ds["uo"] if "uo" in currents_ds.variables else currents_ds.get("u")
        vo_da = currents_ds["vo"] if "vo" in currents_ds.variables else currents_ds.get("v")

        if "depth" in uo_da.dims:
            uo_da = uo_da.isel(depth=0)
        if "depth" in vo_da.dims:
            vo_da = vo_da.isel(depth=0)

        self.grid_lat = uo_da["latitude"].values if "latitude" in uo_da.coords else uo_da["lat"].values
        self.grid_lon = uo_da["longitude"].values if "longitude" in uo_da.coords else uo_da["lon"].values

        self.uo = np.nan_to_num(uo_da.values, nan=0.0)
        self.vo = np.nan_to_num(vo_da.values, nan=0.0)

        if bathymetry_ds is not None and ("elevation" in bathymetry_ds.variables or "z" in bathymetry_ds.variables):
            bath_var = "elevation" if "elevation" in bathymetry_ds.variables else "z"
            bath_da = bathymetry_ds[bath_var]
            bath_lat = bath_da["lat"].values if "lat" in bath_da.coords else bath_da["latitude"].values
            bath_lon = bath_da["lon"].values if "lon" in bath_da.coords else bath_da["longitude"].values

            lat_idx = np.searchsorted(bath_lat, self.grid_lat)
            lon_idx = np.searchsorted(bath_lon, self.grid_lon)

            lat_idx = np.clip(lat_idx, 0, len(bath_lat) - 1)
            lon_idx = np.clip(lon_idx, 0, len(bath_lon) - 1)

            self.depth = bath_da.values[np.ix_(lat_idx, lon_idx)]
        else:
            # Default depth grid (deep water everywhere)
            self.depth = np.full(self.uo.shape, -100.0, dtype=np.float32)

    def _heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> float:
        return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

    def _get_cost(self, current: Tuple[int, int], nxt: Tuple[int, int]) -> float:
        r1, c1 = current
        r2, c2 = nxt

        if self.depth is not None and self.depth[r2, c2] > self.min_depth_m:
            return float("inf")

        distance = math.sqrt((r1 - r2)**2 + (c1 - c2)**2)

        travel_dir = np.array([r2 - r1, c2 - c1], dtype=np.float32)
        norm = np.linalg.norm(travel_dir)
        if norm == 0:
            return float("inf")
        travel_dir = travel_dir / norm

        current_vec = np.array([self.uo[r2, c2], self.vo[r2, c2]], dtype=np.float32)

        alignment = float(np.dot(travel_dir, current_vec))

        if alignment > 0:
            current_cost = self.cost_with_current * (1 - alignment)
        else:
            current_cost = self.cost_against_current * (-alignment)

        return distance * self.cost_per_km + current_cost

    def _smooth_path(self, path: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        if len(path) < 3:
            return path
        smoothed = [path[0]]
        for i in range(1, len(path) - 1):
            prev_dir = (path[i][0] - smoothed[-1][0], path[i][1] - smoothed[-1][1])
            next_dir = (path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1])
            if prev_dir != next_dir:
                smoothed.append(path[i])
        smoothed.append(path[-1])
        return smoothed

    def predict(self, start_lat: float, start_lon: float, target_lat: float, target_lon: float) -> Dict[str, Any]:
        if self.grid_lat is None or self.grid_lon is None:
            # Generate synthetic grid if not loaded
            lats = np.linspace(min(start_lat, target_lat) - 1.0, max(start_lat, target_lat) + 1.0, 50)
            lons = np.linspace(min(start_lon, target_lon) - 1.0, max(start_lon, target_lon) + 1.0, 50)
            self.grid_lat = lats
            self.grid_lon = lons
            self.uo = np.zeros((50, 50), dtype=np.float32)
            self.vo = np.zeros((50, 50), dtype=np.float32)
            self.depth = np.full((50, 50), -100.0, dtype=np.float32)

        start_r = int(np.argmin(np.abs(self.grid_lat - start_lat)))
        start_c = int(np.argmin(np.abs(self.grid_lon - start_lon)))
        target_r = int(np.argmin(np.abs(self.grid_lat - target_lat)))
        target_c = int(np.argmin(np.abs(self.grid_lon - target_lon)))

        start_node = (start_r, start_c)
        target_node = (target_r, target_c)

        open_set = []
        heapq.heappush(open_set, (0, start_node))
        came_from = {}
        g_score = {start_node: 0.0}
        f_score = {start_node: self._heuristic(start_node, target_node)}

        visited = set()

        while open_set:
            _, current = heapq.heappop(open_set)

            if current == target_node:
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()

                path = self._smooth_path(path)
                route_coords = [
                    {"lat": float(self.grid_lat[r]), "lon": float(self.grid_lon[c]), "step": idx}
                    for idx, (r, c) in enumerate(path)
                ]
                return {"route": route_coords, "steps": len(route_coords)}

            visited.add(current)

            r, c = current
            neighbors = [
                (r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1),
                (r - 1, c - 1), (r - 1, c + 1), (r + 1, c - 1), (r + 1, c + 1),
            ]

            for neighbor in neighbors:
                nr, nc = neighbor
                if 0 <= nr < self.uo.shape[0] and 0 <= nc < self.uo.shape[1]:
                    if neighbor in visited:
                        continue

                    cost = self._get_cost(current, neighbor)
                    if cost == float("inf"):
                        continue

                    tentative_g_score = g_score[current] + cost

                    if tentative_g_score < g_score.get(neighbor, float("inf")):
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g_score
                        f_score[neighbor] = tentative_g_score + self._heuristic(neighbor, target_node)
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))

        # Fallback to direct path if no route found
        return {
            "route": [
                {"lat": start_lat, "lon": start_lon, "step": 0},
                {"lat": target_lat, "lon": target_lon, "step": 1},
            ],
            "steps": 2,
        }


class RouteOptimizationService:
    """
    A* fuel-efficient route optimization service.
    Wraps RouteOptimizationModel.
    """

    def __init__(self, model7_instance: Optional[Any] = None):
        if model7_instance is None:
            self.model = RouteOptimizationModel()
        else:
            self.model = model7_instance

    def optimize(
        self,
        start_lat: float,
        start_lon: float,
        dest_lat: float,
        dest_lon: float,
    ) -> Dict[str, Any]:
        """
        Computes the fuel-efficient route from start to destination.
        """
        try:
            result = self.model.predict(start_lat, start_lon, dest_lat, dest_lon)
            route = result.get("route", [])
            steps = result.get("steps", len(route))

            dist_km = _haversine_km(start_lat, start_lon, dest_lat, dest_lon)
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
            return self._straight_line_fallback(start_lat, start_lon, dest_lat, dest_lon, reason=str(e))

    def predict(self, start_lat: float, start_lon: float, dest_lat: float, dest_lon: float) -> Dict[str, Any]:
        """Alias for optimize."""
        return self.optimize(start_lat, start_lon, dest_lat, dest_lon)

    def _straight_line_fallback(
        self,
        start_lat: float, start_lon: float,
        dest_lat: float, dest_lon: float,
        reason: str = "model_not_available",
    ) -> Dict[str, Any]:
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
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(a))
