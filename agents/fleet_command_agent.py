"""
BlueFish AI - Fleet Command Agent (FastAPI-integrated Background Task)
======================================================================
Wires the existing FleetCommandAgent (AGENTS/agent_2.py) to:
  - The model registry (Models 5, 6, 10)
  - Redis Geo-index for live vessel positions
  - Supabase for writing safety_alerts to the DB
  - Redis Pub/Sub for pushing collision alerts

This module exposes `start_fleet_polling_loop()` which runs as an
asyncio background task launched from main.py's startup event.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

# ── Import existing agent code ────────────────────────────────────────────────
_agents_dir = Path(__file__).parent.parent.parent / "AGENTS"
sys.path.insert(0, str(_agents_dir))

from agent_2 import (  # noqa: E402
    FleetCommandAgent,
    Model5Client,
    Model6Client,
    RedisVesselPositionSource,
    StaticVesselPositionSource,
)

logger = logging.getLogger("bluefish.fleet_command_wire")

__all__ = ["start_fleet_polling_loop", "FleetCommandAgent"]


def _build_alert_publisher(redis_sync_client, supabase_client):
    """Returns a callable that publishes collision alerts to Redis AND inserts into Supabase."""

    CHANNEL = "bluefish:safety_alerts"

    def publisher(alert: dict):
        # 1. Redis Pub/Sub (real-time push to connected frontends)
        try:
            redis_sync_client.publish(CHANNEL, json.dumps(alert, default=str))
        except Exception as e:
            logger.error(f"Redis publish failed: {e}")

        # 2. Supabase safety_alerts table (durable record)
        try:
            mmsi1 = alert.get("vessel_1_mmsi", "")
            mmsi2 = alert.get("vessel_2_mmsi", "")
            collision_lat = alert.get("collision_lat", 0.0)
            collision_lon = alert.get("collision_lon", 0.0)

            supabase_client.table("safety_alerts").insert({
                "mmsi": mmsi1,
                "alert_type": "collision",
                "severity": alert.get("severity", "HIGH"),
                "details": alert,
                "location": f"POINT({collision_lon} {collision_lat})",
                "status": "active",
            }).execute()
        except Exception as e:
            logger.error(f"Supabase alert insert failed: {e}")

    return publisher


def _build_compliance_store(supabase_client):
    """Returns a callable that writes compliance flags (MPA intrusions) to Supabase."""

    def store(flag: dict):
        try:
            mmsi = flag.get("mmsi", "")
            flags = flag.get("flags", [])
            severity = "HIGH" if "inside_mpa" in flags else "MEDIUM"

            supabase_client.table("safety_alerts").insert({
                "mmsi": mmsi,
                "alert_type": "mpa_intrusion" if "inside_mpa" in flags else "anomaly",
                "severity": severity,
                "details": flag,
                "status": "active",
            }).execute()
        except Exception as e:
            logger.error(f"Compliance store write failed: {e}")

    return store


def build_fleet_agent(model_registry, redis_sync_client, supabase_client) -> FleetCommandAgent:
    """
    Constructs the FleetCommandAgent wired to all dependencies.
    """
    from core.config import get_settings
    from core.database import is_in_mpa, is_in_eez
    settings = get_settings()

    # Vessel position source
    position_source = RedisVesselPositionSource(
        redis_client=redis_sync_client,
        geo_key=settings.REDIS_GEO_KEY,
        meta_key_prefix=settings.REDIS_META_PREFIX,
    )

    # Model 5 Client
    model5_client: Optional[Model5Client] = None
    if model_registry.model5 is not None:
        model5_client = Model5Client(model_registry.model5)

    # Model 6 Client
    model6_client: Optional[Model6Client] = None
    if model_registry.model6_forest is not None and model_registry.model6_scaler is not None:
        # We need temp pkl paths since Model6Client loads from file
        # Instead, patch the Model6Client directly
        model6_client = _build_model6_client_from_objects(
            model_registry.model6_forest,
            model_registry.model6_scaler,
        )

    alert_publisher = _build_alert_publisher(redis_sync_client, supabase_client)
    compliance_store = _build_compliance_store(supabase_client)

    return FleetCommandAgent(
        position_source=position_source,
        model5_client=model5_client,
        model6_client=model6_client,
        collision_model=model_registry.model10,
        alert_publisher=alert_publisher,
        compliance_store=compliance_store,
        max_vessels_per_cycle=settings.FLEET_MAX_VESSELS_PER_CYCLE,
    )


def _build_model6_client_from_objects(iso_forest, scaler) -> "Model6Client":
    """
    Creates a Model6Client instance directly from pre-loaded objects,
    bypassing the file-path constructor.
    """
    from core.database import is_in_mpa, is_in_eez

    client = object.__new__(Model6Client)
    client.iso_forest = iso_forest
    client.scaler = scaler
    client.mpa_checker = is_in_mpa
    client.eez_checker = is_in_eez
    client.anomaly_score_threshold = -0.1
    return client


async def start_fleet_polling_loop(
    model_registry,
    redis_sync_client,
    supabase_client,
    poll_interval_seconds: int = 60,
):
    """
    Async background task that runs the Fleet Command Agent in a continuous loop.
    Launched from main.py's lifespan startup handler.

    Uses asyncio.to_thread() to run the sync polling cycle without blocking
    the main event loop (critical for FastAPI to remain responsive).
    """
    logger.info(f"Fleet polling loop starting. Interval: {poll_interval_seconds}s")
    agent = build_fleet_agent(model_registry, redis_sync_client, supabase_client)

    while True:
        try:
            logger.debug("Fleet cycle starting...")
            report = await asyncio.to_thread(agent.run_cycle)
            logger.info(
                f"Fleet cycle complete. "
                f"vessels={report.vessels_checked} "
                f"degraded={report.degraded} "
                f"reasons={report.degraded_reasons}"
            )
        except asyncio.CancelledError:
            logger.info("Fleet polling loop cancelled — shutting down gracefully.")
            break
        except Exception as e:
            logger.error(f"Fleet cycle unhandled error: {e}", exc_info=True)

        await asyncio.sleep(poll_interval_seconds)
