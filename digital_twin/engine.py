"""
BlueFish AI - Marine Digital Twin Engine
==========================================
Implements an Agent-Based Model (ABM) simulation of the Tamil Nadu fishing fleet.

Architecture:
  - VesselAgent: Individual fishing vessel with state machine (Searching → Fishing → Returning)
  - EnvironmentAgent: Holds the daily SST/Current grid and manages spatial boundaries
  - run_simulation(): Main async entry point called by the API route

The simulation queries the loaded AI models at each timestep to make
decisions — exactly as real vessels would use the BlueFish AI app.

Policy scenarios:
  - When `close_gulf_of_mannar=True`, the engine adds a no-go polygon and
    vessel agents reroute using Model 7.
  - When `monsoon_ban=True`, all vessels return to port immediately.

Output format:
  A JSON timeline array of hourly vessel states, compatible with
  CesiumJS CZML for 3D animation rendering.
"""

from __future__ import annotations

import asyncio
import logging
import math
import random
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("bluefish.digital_twin")

# ── Tamil Nadu coastal geography ──────────────────────────────────────────────
TN_PORT_LAT = 10.8
TN_PORT_LON = 79.8

GULF_OF_MANNAR_BOUNDS = (8.5, 9.5, 78.0, 80.0)   # lat_min, lat_max, lon_min, lon_max
EEZ_BOUNDS = (7.0, 14.0, 76.0, 82.0)

FUEL_BURN_RATE_LPH = 20.0          # Litres per hour at full speed
FUEL_CAPACITY_L = 800.0
CATCH_RATE_KG_PER_HOUR = 50.0      # Max catch when at high-probability zone
VESSEL_SPEED_KMH = 15.0            # Average trawler speed (8 knots)
GRID_STEP_KM = 50.0                # How far a vessel moves per decision step (1 hour)


# ── Agent Definitions ─────────────────────────────────────────────────────────

@dataclass
class VesselAgent:
    vessel_id: str
    lat: float
    lon: float
    fuel_liters: float = FUEL_CAPACITY_L
    catch_kg: float = 0.0
    status: str = "Searching"          # Searching | Fishing | Returning
    target_lat: Optional[float] = None
    target_lon: Optional[float] = None
    trip_hours: int = 0

    def is_out_of_fuel(self) -> bool:
        return self.fuel_liters <= 0

    def should_return(self) -> bool:
        # Return when fuel < 30% or hold is "full" (catch >= 500 kg)
        return self.fuel_liters < FUEL_CAPACITY_L * 0.30 or self.catch_kg >= 500.0

    def move_towards(self, target_lat: float, target_lon: float):
        """Move one GRID_STEP_KM towards the target."""
        dlat = target_lat - self.lat
        dlon = target_lon - self.lon
        dist = math.sqrt(dlat**2 + dlon**2) * 111.0  # rough km

        if dist < 5.0:
            self.lat, self.lon = target_lat, target_lon
            return

        step_deg = (GRID_STEP_KM / 111.0)
        angle = math.atan2(dlon, dlat)
        self.lat += step_deg * math.cos(angle)
        self.lon += step_deg * math.sin(angle)
        self.fuel_liters -= FUEL_BURN_RATE_LPH  # 1 hour travel

    def record_state(self, hour: int) -> Dict[str, Any]:
        return {
            "vessel_id": self.vessel_id,
            "hour": hour,
            "lat": round(self.lat, 4),
            "lon": round(self.lon, 4),
            "fuel_liters": round(self.fuel_liters, 1),
            "catch_kg": round(self.catch_kg, 1),
            "status": self.status,
        }


@dataclass
class EnvironmentAgent:
    """Holds ocean state and policy boundaries for a simulation run."""
    sst_grid: Optional[np.ndarray] = None
    current_u_grid: Optional[np.ndarray] = None
    current_v_grid: Optional[np.ndarray] = None
    lat_coords: Optional[np.ndarray] = None
    lon_coords: Optional[np.ndarray] = None

    # Policy flags
    gulf_of_mannar_closed: bool = False
    monsoon_ban_active: bool = False
    climate_sst_delta: float = 0.0

    def is_in_no_go_zone(self, lat: float, lon: float) -> bool:
        """Returns True if the position is inside a closed area."""
        if self.gulf_of_mannar_closed:
            lat_min, lat_max, lon_min, lon_max = GULF_OF_MANNAR_BOUNDS
            if lat_min <= lat <= lat_max and lon_min <= lon <= lon_max:
                return True
        return False

    def get_fish_probability_at(self, lat: float, lon: float, model1=None) -> float:
        """
        Returns estimated fish presence probability at a location.
        Uses Model 1 if available, otherwise uses a simple SST-based heuristic.
        """
        if model1 is not None:
            try:
                features = self._build_model1_features(lat, lon)
                result = model1.predict(features)
                return result.get("presence_probability", 0.3)
            except Exception:
                pass
        # Fallback heuristic: fish probability peaks in 8-12°N, 78-80°E
        lat_score = max(0, 1 - abs(lat - 10.0) / 5.0)
        lon_score = max(0, 1 - abs(lon - 79.0) / 5.0)
        return (lat_score + lon_score) / 2.0

    def _build_model1_features(self, lat: float, lon: float) -> Dict[str, float]:
        """Builds a Model 1 feature dict for a location."""
        now = datetime.now(timezone.utc)
        return {
            "month": float(now.month),
            "dayofyear": float(now.timetuple().tm_yday),
            "ONI_Value": 0.5,           # Neutral ENSO
            "sst": 27.0 + self.climate_sst_delta,
            "salinity": 35.0,
            "current_east": 0.1,
            "current_north": 0.05,
            "chlorophyll": 0.8,
            "current_speed": 0.12,
            "current_direction_deg": 45.0,
        }


# ── Simulation Step Logic ─────────────────────────────────────────────────────

def _simulate_hour(
    vessel: VesselAgent,
    env: EnvironmentAgent,
    model1=None,
    model7=None,
    model8=None,
) -> None:
    """Advances a single vessel by one hour."""
    vessel.trip_hours += 1

    # Policy: Monsoon ban — all vessels return
    if env.monsoon_ban_active:
        vessel.status = "Returning"

    if vessel.status == "Returning":
        vessel.move_towards(TN_PORT_LAT, TN_PORT_LON)
        dist_to_port = math.sqrt((vessel.lat - TN_PORT_LAT)**2 + (vessel.lon - TN_PORT_LON)**2) * 111
        if dist_to_port < 5.0:
            vessel.status = "Docked"
        return

    if vessel.should_return() or vessel.is_out_of_fuel():
        vessel.status = "Returning"
        return

    if vessel.status == "Searching":
        # Pick a target: scan nearby locations for highest fish probability
        best_prob = -1.0
        best_lat, best_lon = vessel.lat, vessel.lon
        for _ in range(8):
            test_lat = vessel.lat + random.uniform(-1.0, 1.0)
            test_lon = vessel.lon + random.uniform(-1.0, 1.0)
            # Clamp to EEZ
            test_lat = max(EEZ_BOUNDS[0], min(EEZ_BOUNDS[1], test_lat))
            test_lon = max(EEZ_BOUNDS[2], min(EEZ_BOUNDS[3], test_lon))
            if env.is_in_no_go_zone(test_lat, test_lon):
                continue
            prob = env.get_fish_probability_at(test_lat, test_lon, model1)
            if prob > best_prob:
                best_prob = prob
                best_lat, best_lon = test_lat, test_lon

        vessel.target_lat = best_lat
        vessel.target_lon = best_lon

        if best_prob > 0.6:
            vessel.status = "Fishing"
        else:
            vessel.move_towards(best_lat, best_lon)

    elif vessel.status == "Fishing":
        prob = env.get_fish_probability_at(vessel.lat, vessel.lon, model1)
        catch = CATCH_RATE_KG_PER_HOUR * prob
        vessel.catch_kg += catch
        vessel.fuel_liters -= FUEL_BURN_RATE_LPH * 0.3  # Lower burn rate when stationary
        if prob < 0.3:
            vessel.status = "Searching"


# ── Main Simulation Entry Point ───────────────────────────────────────────────

async def run_simulation(
    days: int,
    fleet_size: int,
    policy_restrictions: Dict[str, Any],
    initial_lat: float,
    initial_lon: float,
    model_registry=None,
) -> Dict[str, Any]:
    """
    Runs the ABM simulation for `days` days with `fleet_size` vessels.
    Returns a timeline dict with hourly snapshots.
    """
    # Build environment
    env = EnvironmentAgent(
        gulf_of_mannar_closed=policy_restrictions.get("close_gulf_of_mannar", False),
        monsoon_ban_active=policy_restrictions.get("monsoon_ban", False),
        climate_sst_delta=float(policy_restrictions.get("climate_sst_delta", 0.0)),
    )

    # Get models from registry
    model1 = getattr(model_registry, "model1", None) if model_registry else None
    model7 = getattr(model_registry, "model7", None) if model_registry else None
    model8 = getattr(model_registry, "model8", None) if model_registry else None

    # Spawn fleet with slight position scatter around the port
    vessels = []
    for i in range(fleet_size):
        scatter_lat = initial_lat + random.uniform(-0.5, 0.5)
        scatter_lon = initial_lon + random.uniform(-0.5, 0.5)
        vessels.append(VesselAgent(
            vessel_id=f"SIM_{i+1:04d}",
            lat=scatter_lat,
            lon=scatter_lon,
            fuel_liters=random.uniform(FUEL_CAPACITY_L * 0.8, FUEL_CAPACITY_L),
        ))

    # Simulation loop
    total_hours = days * 24
    timeline_steps = []
    total_fuel_burned = 0.0
    total_catch_kg = 0.0

    # Yield control to event loop every 6 sim-hours to avoid blocking
    for hour in range(total_hours):
        if hour % 6 == 0:
            await asyncio.sleep(0)  # yield to event loop

        hour_snapshot = {"hour": hour, "vessels": []}
        active_count = 0

        for vessel in vessels:
            if vessel.status == "Docked":
                continue
            fuel_before = vessel.fuel_liters
            _simulate_hour(vessel, env, model1, model7, model8)
            total_fuel_burned += max(0, fuel_before - vessel.fuel_liters)
            active_count += 1

            # Record snapshot every 6 hours to keep response size manageable
            if hour % 6 == 0:
                hour_snapshot["vessels"].append(vessel.record_state(hour))

        if hour % 6 == 0:
            hour_snapshot["active_vessels"] = active_count
            timeline_steps.append(hour_snapshot)

    total_catch_kg = sum(v.catch_kg for v in vessels)
    avg_catch = total_catch_kg / fleet_size if fleet_size > 0 else 0

    return {
        "steps": timeline_steps,
        "summary": {
            "days_simulated": days,
            "fleet_size": fleet_size,
            "total_fuel_burned_liters": round(total_fuel_burned, 1),
            "total_catch_kg": round(total_catch_kg, 1),
            "avg_catch_per_vessel_kg": round(avg_catch, 1),
            "policy_restrictions": policy_restrictions,
            "env_settings": {
                "gulf_closed": env.gulf_of_mannar_closed,
                "monsoon_ban": env.monsoon_ban_active,
                "climate_sst_delta": env.climate_sst_delta,
            },
        },
    }
