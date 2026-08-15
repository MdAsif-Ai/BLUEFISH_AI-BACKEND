"""
BlueFish AI - Data Ingestion Agent (FastAPI-integrated)
=======================================================
Wires the existing DataQualityAgent (AGENTS/agent_3.py) to:
  - Supabase Storage (for quarantine uploads)
  - Supabase DB (for audit log writes)
  - Multi-threaded processing pipeline
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

_agents_dir = Path(__file__).parent.parent.parent / "AGENTS"
sys.path.insert(0, str(_agents_dir))

from agent_3 import DataQualityAgent, AuditReport  # noqa: E402

logger = logging.getLogger("bluefish.data_ingestion_wire")

__all__ = ["DataQualityAgent", "AuditReport", "run_data_ingestion"]

# ── Default source configuration for known satellite data types ───────────────
DEFAULT_SOURCE_CONFIG = {
    "*.nc": {
        "type": "netcdf",
        "expected_variables": ["sst", "zos", "uo", "vo", "so", "chl"],
    },
    "sst_*.nc": {
        "type": "netcdf",
        "expected_variables": ["sst"],
        "required_date_start": "2023-01-01",
    },
    "tide_*.csv": {
        "type": "csv",
        "expected_columns": ["datetime", "tide_height", "station"],
        "content_signature_key": "tide",
    },
    "vms_*.parquet": {
        "type": "parquet",
        "expected_columns": ["mmsi", "lat", "lon", "timestamp", "speed", "heading"],
        "date_column": "timestamp",
    },
}


def run_data_ingestion(
    filepaths: List[str],
    supabase_client: Any,
    quarantine_bucket: str = "quarantine",
    max_workers: int = 4,
) -> AuditReport:
    """
    Runs the multi-threaded data quality pipeline on a list of files.

    Failed files are:
      1. Physically moved to the local quarantine/ directory
      2. Uploaded to Supabase Storage quarantine bucket
      3. Logged to the data_audit_log table in Supabase

    Returns the full AuditReport.
    """
    import tempfile
    import os
    import shutil

    local_quarantine_dir = Path(__file__).parent.parent / "quarantine"
    local_quarantine_dir.mkdir(exist_ok=True)

    agent = DataQualityAgent(
        source_config=DEFAULT_SOURCE_CONFIG,
        max_workers=max_workers,
    )

    report = agent.run_audit(
        filepaths=filepaths,
        quarantine_dir=str(local_quarantine_dir),
        supabase_client=supabase_client,
    )

    # Upload quarantined files to Supabase Storage
    for failed_result in report.failed_files():
        quarantine_path = Path(failed_result.filepath)
        if quarantine_path.exists():
            try:
                storage_path = f"quarantine/{quarantine_path.name}"
                with open(quarantine_path, "rb") as f:
                    supabase_client.storage.from_(quarantine_bucket).upload(
                        storage_path, f, {"upsert": "true"}
                    )
                logger.warning(f"Quarantined to Supabase Storage: {storage_path}")
            except Exception as e:
                logger.error(f"Failed to upload quarantined file to Supabase: {e}")

    return report
