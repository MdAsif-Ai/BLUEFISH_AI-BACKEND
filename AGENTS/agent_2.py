"""
BlueFish AI - Fleet Command, Safety & Compliance Agent (Production)
=======================================================================
Fleet-wide, real-time monitoring - NOT a per-user conversational agent.
Orchestrates Models 5 (Fleet Density), 6 (Anomaly/Compliance), and 10
(Collision Risk) on a continuous polling cycle over live vessel positions.

Design principles applied here:
  - This agent is a POLLING LOOP, not a request/response handler like the
    Advisory Agent - it runs continuously against live fleet data, not
    once per user question.
  - Model 10 (collision) is genuinely time-critical - its alerts are
    pushed immediately via Pub/Sub, not batched with the others.
  - Model 5 (density) and Model 6 (anomaly/compliance) are lower urgency -
    aggregated into a periodic Command Center report rather than pushed
    per-detection, to avoid alert fatigue on non-urgent findings.
  - Compliance flags (Model 6) are written to durable storage (Postgres),
    never cache-only - these may feed real enforcement action later and
    cannot be allowed to silently expire from a cache TTL.
  - One vessel-fetch failure or one model failure does not stop the whole
    polling cycle - each stage degrades independently and logs clearly.

IMPORTANT GAP FLAGGED: no Model 6 service class exists in the codebase
yet (only isolation_forest.pkl + scaler.pkl artifacts) - Model2/5/7/8/10
all have proper service wrappers, Model 6 does not. The Model6Client
below is written against the ARCHITECTURE SPEC (VMS behavioral features +
EEZ/MPA point-in-polygon checks feeding an Isolation Forest), but it has
NOT been validated against however your team actually engineered Model
6's training features - the feature order/composition below is a
best-effort reconstruction and MUST be checked against the real training
pipeline before this ships, or scores will be meaningless (Isolation
Forest is extremely sensitive to feature order/scaling matching training
exactly).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pickle

logger = logging.getLogger("bluefish.fleet_command_agent")
logger.setLevel(logging.INFO)


# ======================================================================
# Result containers
# ======================================================================

@dataclass
class StageResult:
    stage_name: str
    ok: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    latency_ms: float = 0.0


@dataclass
class CycleReport:
    timestamp: str
    vessels_checked: int
    density: Optional[StageResult] = None
    anomalies: Optional[StageResult] = None
    collisions: Optional[StageResult] = None
    degraded: bool = False
    degraded_reasons: List[str] = field(default_factory=list)


# ======================================================================
# Live vessel position source - abstracted so this works against a Redis
# geospatial store in production and a static list in tests.
# ======================================================================

class VesselPositionSource:
    def get_active_vessels(self) -> List[Dict[str, Any]]:
        """
        Must return a list of dicts, each with at minimum:
        mmsi, lat, lon, speed (knots), heading (degrees), timestamp
        """
        raise NotImplementedError


class StaticVesselPositionSource(VesselPositionSource):
    """Test/dev fallback - wraps a fixed list."""

    def __init__(self, vessels: List[Dict[str, Any]]):
        self._vessels = vessels

    def get_active_vessels(self) -> List[Dict[str, Any]]:
        return self._vessels


class RedisVesselPositionSource(VesselPositionSource):
    """
    Production source - reads live positions from Redis geospatial keys.
    Requires the `redis` package and assumes positions are written there
    by the ingestion layer (e.g. GEOADD "fleet:live" lon lat mmsi, plus a
    companion hash per mmsi holding speed/heading/timestamp).
    """

    def __init__(self, redis_client, geo_key: str = "fleet:live", meta_key_prefix: str = "fleet:meta:"):
        self._r = redis_client
        self._geo_key = geo_key
        self._meta_prefix = meta_key_prefix

    def get_active_vessels(self) -> List[Dict[str, Any]]:
        import json
        members = self._r.zrange(self._geo_key, 0, -1)
        vessels = []
        for mmsi_bytes in members:
            mmsi = mmsi_bytes.decode() if isinstance(mmsi_bytes, bytes) else str(mmsi_bytes)
            pos = self._r.geopos(self._geo_key, mmsi)
            meta_raw = self._r.get(f"{self._meta_prefix}{mmsi}")
            if not pos or pos[0] is None or meta_raw is None:
                continue
            lon, lat = pos[0]
            meta = json.loads(meta_raw)
            vessels.append({
                "mmsi": mmsi,
                "lat": float(lat),
                "lon": float(lon),
                "speed": float(meta.get("speed", 0.0)),
                "heading": float(meta.get("heading", 0.0)),
                "timestamp": meta.get("timestamp"),
            })
        return vessels


# ======================================================================
# Model 5 wrapper - thin pass-through to FleetDensityModel, but takes
# live vessel dicts directly rather than requiring a pre-built dataframe,
# since the real-time path shouldn't force a parquet round-trip.
# ======================================================================

class Model5Client:
    def __init__(self, fleet_density_model):
        self.model = fleet_density_model

    def predict(self, vessels: List[Dict[str, Any]], target_date: str) -> Dict[str, Any]:
        if not vessels:
            return {"zones": [], "boats": []}
        df = pd.DataFrame(vessels)
        # FleetDensityModel expects cell_ll_lat/cell_ll_lon/date/fishing_hours/mmsi
        df = df.rename(columns={"lat": "cell_ll_lat", "lon": "cell_ll_lon"})
        df["date"] = target_date
        if "fishing_hours" not in df.columns:
            # Live positions don't carry a fishing_hours field the way
            # historical VMS data does - treat every currently-active
            # vessel as in-scope for density clustering (fishing_hours>0
            # filter in the model is a HISTORICAL-DATA convention, not
            # applicable to a live position feed).
            df["fishing_hours"] = 1.0
        return self.model.predict(df, target_date)


# ======================================================================
# Model 6 wrapper - RECONSTRUCTED FROM SPEC, NOT FROM AN EXISTING SERVICE
# CLASS. Flagged prominently above and here again: verify feature
# composition/order against the real training pipeline before trusting
# scores from this in production.
# ======================================================================

class Model6Client:
    """
    Isolation Forest anomaly/compliance detector.

    Expected feature vector (per vessel, per polling cycle) - VERIFY this
    matches training:
        [time_since_last_report_minutes,
         speed_deviation_from_own_baseline,
         inside_mpa (0/1),
         inside_eez (0/1, 0 = outside home EEZ / in another nation's waters),
         effort_zscore_vs_own_history]

    `mpa_checker` and `eez_checker` are injected callables (lat, lon) ->
    bool, backed by PostGIS point-in-polygon queries in production (see
    architecture doc's PostgreSQL+PostGIS layer) - this class does not
    implement geospatial boundary logic itself.
    """

    def __init__(self, isolation_forest_path: str, scaler_path: str,
                 mpa_checker=None, eez_checker=None, anomaly_score_threshold: float = -0.1):
        with open(isolation_forest_path, "rb") as f:
            self.iso_forest = pickle.load(f)
        with open(scaler_path, "rb") as f:
            self.scaler = pickle.load(f)
        self.mpa_checker = mpa_checker
        self.eez_checker = eez_checker
        self.anomaly_score_threshold = anomaly_score_threshold

    def _build_feature_vector(self, vessel: Dict[str, Any], vessel_history: Optional[Dict[str, Any]]) -> List[float]:
        time_since_last = vessel.get("minutes_since_last_report", 0.0)

        baseline_speed = (vessel_history or {}).get("avg_speed", vessel.get("speed", 0.0))
        speed_dev = abs(vessel.get("speed", 0.0) - baseline_speed)

        inside_mpa = 0.0
        if self.mpa_checker is not None:
            inside_mpa = 1.0 if self.mpa_checker(vessel["lat"], vessel["lon"]) else 0.0

        inside_eez = 1.0
        if self.eez_checker is not None:
            inside_eez = 1.0 if self.eez_checker(vessel["lat"], vessel["lon"]) else 0.0

        effort_zscore = (vessel_history or {}).get("effort_zscore", 0.0)

        return [time_since_last, speed_dev, inside_mpa, inside_eez, effort_zscore]

    def predict(self, vessel: Dict[str, Any], vessel_history: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        features = self._build_feature_vector(vessel, vessel_history)
        x = self.scaler.transform([features])
        score = float(self.iso_forest.decision_function(x)[0])
        is_anomaly = score < self.anomaly_score_threshold

        reasons = []
        if features[2] == 1.0:
            reasons.append("inside_mpa")
        if features[3] == 0.0:
            reasons.append("outside_home_eez")
        if features[1] > 5.0:
            reasons.append("speed_deviation")

        return {
            "mmsi": vessel.get("mmsi"),
            "anomaly_score": round(score, 4),
            "is_anomaly": is_anomaly,
            "flags": reasons,
        }


# ======================================================================
# THE AGENT
# ======================================================================

class FleetCommandAgent:
    """
    Runs one monitoring cycle across the full live fleet. Call
    `run_cycle()` on a schedule (e.g. every 30-60s via your orchestrator)
    or wrap it in a simple `while True` loop with a sleep for a
    standalone service.
    """

    def __init__(
        self,
        position_source: VesselPositionSource,
        model5_client: Optional[Model5Client] = None,
        model6_client: Optional[Model6Client] = None,
        collision_model=None,   # CollisionDetectionModel instance
        alert_publisher=None,   # callable(alert: dict) -> None, e.g. Redis Pub/Sub publish
        compliance_store=None,  # callable(flag: dict) -> None, e.g. Postgres insert
        max_vessels_per_cycle: int = 20000,
    ):
        self.position_source = position_source
        self.model5 = model5_client
        self.model6 = model6_client
        self.model10 = collision_model
        self.alert_publisher = alert_publisher
        self.compliance_store = compliance_store
        self.max_vessels_per_cycle = max_vessels_per_cycle

    def _safe_call(self, name: str, fn, *args, **kwargs) -> StageResult:
        start = time.monotonic()
        try:
            data = fn(*args, **kwargs)
            latency = (time.monotonic() - start) * 1000
            logger.info(f"stage_success stage={name} latency_ms={latency:.1f}")
            return StageResult(stage_name=name, ok=True, data=data, latency_ms=latency)
        except Exception as e:
            latency = (time.monotonic() - start) * 1000
            logger.error(f"stage_failed stage={name} error={e} latency_ms={latency:.1f}", exc_info=True)
            return StageResult(stage_name=name, ok=False, error=str(e), latency_ms=latency)

    def run_cycle(self, target_date: Optional[str] = None) -> CycleReport:
        target_date = target_date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
        report = CycleReport(timestamp=datetime.now(timezone.utc).isoformat(), vessels_checked=0)

        # --- Fetch live positions ---
        try:
            vessels = self.position_source.get_active_vessels()
        except Exception as e:
            logger.error(f"vessel_fetch_failed error={e}", exc_info=True)
            report.degraded = True
            report.degraded_reasons.append(f"vessel position fetch failed: {e}")
            return report  # nothing else can run without positions

        if len(vessels) > self.max_vessels_per_cycle:
            logger.warning(
                f"vessel_count_exceeds_cap count={len(vessels)} cap={self.max_vessels_per_cycle} "
                f"- truncating this cycle, investigate whether the fleet has genuinely grown "
                f"or a stale-position cleanup job has stopped running upstream"
            )
            vessels = vessels[: self.max_vessels_per_cycle]

        report.vessels_checked = len(vessels)

        if not vessels:
            report.degraded = True
            report.degraded_reasons.append("no active vessels returned this cycle")
            return report

        # --- Model 10: Collision risk - HIGH URGENCY, push immediately ---
        if self.model10 is not None:
            report.collisions = self._safe_call("model10_collision", self.model10.predict, vessels)
            if report.collisions.ok and self.alert_publisher is not None:
                alerts = report.collisions.data.get("alerts", [])
                for alert in alerts:
                    try:
                        self.alert_publisher({"type": "collision_risk", **alert, "cycle_timestamp": report.timestamp})
                    except Exception as e:
                        logger.error(f"alert_publish_failed alert={alert} error={e}", exc_info=True)
        else:
            report.collisions = StageResult(stage_name="model10_collision", ok=False, error="Model 10 not configured")

        # --- Model 5: Fleet density - lower urgency, aggregated in report ---
        if self.model5 is not None:
            report.density = self._safe_call("model5_density", self.model5.predict, vessels, target_date)
        else:
            report.density = StageResult(stage_name="model5_density", ok=False, error="Model 5 not configured")

        # --- Model 6: Anomaly/compliance - per-vessel, durable-store writes ---
        if self.model6 is not None:
            anomaly_results = []
            failures = 0
            for v in vessels:
                result = self._safe_call(f"model6_anomaly_{v.get('mmsi', 'unknown')}", self.model6.predict, v)
                if result.ok:
                    anomaly_results.append(result.data)
                    if result.data.get("is_anomaly") and self.compliance_store is not None:
                        try:
                            self.compliance_store({**result.data, "cycle_timestamp": report.timestamp})
                        except Exception as e:
                            logger.error(f"compliance_store_failed mmsi={v.get('mmsi')} error={e}", exc_info=True)
                else:
                    failures += 1
            report.anomalies = StageResult(
                stage_name="model6_anomaly",
                ok=failures < len(vessels),  # ok if at least some succeeded
                data={"results": anomaly_results, "failures": failures, "total": len(vessels)},
                error=(f"{failures}/{len(vessels)} vessels failed anomaly scoring" if failures else None),
            )
        else:
            report.anomalies = StageResult(stage_name="model6_anomaly", ok=False, error="Model 6 not configured")

        # --- Roll up degradation ---
        for result in [report.density, report.anomalies, report.collisions]:
            if result is not None and not result.ok and result.error and "not configured" not in result.error:
                report.degraded = True
                report.degraded_reasons.append(f"{result.stage_name}: {result.error}")

        return report

