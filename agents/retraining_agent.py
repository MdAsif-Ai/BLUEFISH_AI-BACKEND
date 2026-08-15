"""
BlueFish AI - Retraining Agent (FastAPI-integrated)
====================================================
Wires the existing ContinuousRetrainingAgent (AGENTS/agent_4.py) to
the Supabase client and exposes:
  - check_and_trigger(): Checks if enough new feedback exists to retrain
  - promote_model(): Human-gated promotion API handler (called from the dashboard)
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional

_agents_dir = Path(__file__).parent.parent.parent / "AGENTS"
sys.path.insert(0, str(_agents_dir))

from agent_4 import (  # noqa: E402
    ContinuousRetrainingAgent,
    SupabaseModelRegistry,
    ModelVersion,
)

logger = logging.getLogger("bluefish.retraining_wire")

__all__ = ["ContinuousRetrainingAgent", "SupabaseModelRegistry", "build_retraining_agent"]


def build_retraining_agent(supabase_client: Any) -> ContinuousRetrainingAgent:
    """Constructs the retraining agent wired to Supabase."""
    registry = SupabaseModelRegistry(supabase_client)
    return ContinuousRetrainingAgent(
        registry=registry,
        supabase_client=supabase_client,
        storage_bucket="ml-models",
    )


async def check_and_trigger_retraining(
    supabase_client: Any,
    model_name: str = "model1_pfz",
    min_new_rows: int = 200,
    training_script_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Checks if enough new trip_feedback rows have been collected since the
    last training run. If yes, triggers a training subprocess.

    Returns a status dict for the API response.
    """
    import asyncio

    try:
        # Count new feedback rows
        result = supabase_client.table("trip_feedback").select("id", count="exact").execute()
        total_rows = result.count or 0

        logger.info(f"trip_feedback rows: {total_rows} (threshold: {min_new_rows})")

        if total_rows < min_new_rows:
            return {
                "triggered": False,
                "reason": f"Insufficient feedback data: {total_rows}/{min_new_rows} rows",
                "feedback_count": total_rows,
            }

        if training_script_path is None:
            return {
                "triggered": False,
                "reason": "No training script path configured. Set TRAINING_SCRIPT_PATH env var.",
                "feedback_count": total_rows,
            }

        agent = build_retraining_agent(supabase_client)
        success = await asyncio.to_thread(
            agent.trigger_training_job, model_name, training_script_path
        )

        return {
            "triggered": True,
            "success": success,
            "model_name": model_name,
            "feedback_count": total_rows,
            "note": "New model will appear in model_registry as 'staging'. Human promotion required.",
        }

    except Exception as e:
        logger.error(f"Retraining check failed: {e}", exc_info=True)
        return {"triggered": False, "error": str(e)}
