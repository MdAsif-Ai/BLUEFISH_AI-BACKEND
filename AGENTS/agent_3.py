"""
BlueFish AI - Data Quality & Ingestion Agent (Production-Optimized)
=======================================================================
Validates incoming data files before they reach training or production.
Optimized for production with:
  - Multi-threaded file checking for massive speed improvements.
  - Smart checksumming (skips 15GB files to save CPU).
  - Automated physical quarantine of corrupted files.
  - Supabase database logging for permanent audit trails.
"""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("bluefish.data_quality_agent")
logger.setLevel(logging.INFO)

# ======================================================================
# Result containers
# ======================================================================

@dataclass
class FileCheckResult:
    filepath: str
    file_type: str
    passed: bool
    checks: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    needs_human_review: bool = False
    review_reason: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Converts to dictionary for database insertion."""
        return {
            "filepath": self.filepath,
            "file_type": self.file_type,
            "passed": self.passed,
            "errors": "; ".join(self.errors) if self.errors else None,
            "needs_review": self.needs_human_review,
            "review_reason": self.review_reason,
            "checks": self.checks
        }

@dataclass
class AuditReport:
    run_timestamp: str
    results: List[FileCheckResult] = field(default_factory=list)

    def passed_files(self) -> List[FileCheckResult]: return [r for r in self.results if r.passed]
    def failed_files(self) -> List[FileCheckResult]: return [r for r in self.results if not r.passed]
    def needs_review_files(self) -> List[FileCheckResult]: return [r for r in self.results if r.needs_human_review]

# ======================================================================
# Physical plausibility bounds
# ======================================================================
PLAUSIBILITY_BOUNDS = {
    "sst": (15.0, 35.0), "thetao": (15.0, 35.0),
    "salinity": (30.0, 40.0), "so": (30.0, 40.0),
    "chl": (0.0, 50.0), "chlorophyll": (0.0, 50.0),
    "o2": (0.0, 400.0), "dissolved_oxygen": (0.0, 400.0),
    "zos": (-2.0, 2.0), "ssh": (-2.0, 2.0),
    "elevation": (-11000.0, 9000.0),
}

KNOWN_VARIABLE_SIGNATURES = {
    "tide": {"expected_any_of": ["tide_height", "z", "h"], "red_flag_if_present": ["VHM0", "VMDR", "VTPK", "VCMX"]},
    "wave": {"expected_any_of": ["VHM0", "VMDR", "VTPK", "VCMX"]},
}

# ======================================================================
# Per-file-type check functions
# ======================================================================

def check_netcdf(filepath: str, required_date_start: Optional[str] = None,
                  required_date_end: Optional[str] = None,
                  expected_variables: Optional[List[str]] = None) -> Tuple[Dict[str, Any], List[str]]:
    import xarray as xr
    checks: Dict[str, Any] = {}
    errors: List[str] = []

    file_size_mb = os.path.getsize(filepath) / (1024 * 1024)
    checks["file_size_mb"] = round(file_size_mb, 3)

    if file_size_mb < 0.01:
        errors.append(f"File is suspiciously small ({file_size_mb*1024:.1f} KB) - likely a failed download")

    try:
        ds = xr.open_dataset(filepath)
    except Exception as e:
        errors.append(f"Failed to open as NetCDF: {e}")
        checks["opens_successfully"] = False
        return checks, errors

    checks["opens_successfully"] = True
    checks["variables"] = list(ds.data_vars)

    if expected_variables:
        missing_vars = [v for v in expected_variables if v not in ds.data_vars]
        if missing_vars:
            errors.append(f"Expected variable(s) not found: {missing_vars}")

    if "time" in ds.coords:
        time_min = pd.Timestamp(ds.time.min().values)
        time_max = pd.Timestamp(ds.time.max().values)
        checks["date_range"] = f"{time_min.date()} to {time_max.date()}"

        if required_date_start and time_min > pd.Timestamp(required_date_start):
            errors.append(f"Coverage starts {time_min.date()}, required start is {required_date_start}")
        if required_date_end and time_max < pd.Timestamp(required_date_end):
            errors.append(f"Coverage ends {time_max.date()}, required end is {required_date_end}")

    for var in ds.data_vars:
        try:
            arr = ds[var].values
            if arr.size == 0 or arr.dtype.kind not in "fc": continue
            nan_frac = float(np.isnan(arr).mean())
            checks.setdefault("nan_fractions", {})[var] = round(nan_frac, 4)
            if nan_frac > 0.60:
                errors.append(f"Variable '{var}' has {nan_frac*100:.1f}% NaN - unusually high.")
            if var in PLAUSIBILITY_BOUNDS:
                lo, hi = PLAUSIBILITY_BOUNDS[var]
                valid = arr[~np.isnan(arr)]
                if valid.size > 0 and (valid.min() < lo or valid.max() > hi):
                    errors.append(f"'{var}' range [{valid.min():.2f}, {valid.max():.2f}] outside bounds [{lo}, {hi}]")
        except Exception as e:
            errors.append(f"Could not evaluate variable '{var}': {e}")

    ds.close()
    return checks, errors

def check_csv(filepath: str, expected_columns: Optional[List[str]] = None,
              content_signature_key: Optional[str] = None) -> Tuple[Dict[str, Any], List[str]]:
    checks: Dict[str, Any] = {}
    errors: List[str] = []
    checks["file_size_mb"] = round(os.path.getsize(filepath) / (1024 * 1024), 3)

    try:
        df_sample = pd.read_csv(filepath, nrows=1000)
    except Exception as e:
        errors.append(f"Failed to read as CSV: {e}")
        return checks, errors

    checks["columns"] = list(df_sample.columns)
    if len(df_sample) == 0: errors.append("File opens but contains zero data rows")

    if expected_columns:
        missing = [c for c in expected_columns if c not in df_sample.columns]
        if missing: errors.append(f"Expected column(s) not found: {missing}")

    if content_signature_key and content_signature_key in KNOWN_VARIABLE_SIGNATURES:
        sig = KNOWN_VARIABLE_SIGNATURES[content_signature_key]
        found_red_flags = [c for c in sig.get("red_flag_if_present", []) if c in df_sample.columns]
        if found_red_flags:
            errors.append(f"Filename claims '{content_signature_key}' but contains wave data columns {found_red_flags}")

    return checks, errors

def check_parquet(filepath: str, expected_columns: Optional[List[str]] = None,
                   date_column: Optional[str] = None) -> Tuple[Dict[str, Any], List[str]]:
    checks: Dict[str, Any] = {}
    errors: List[str] = []

    try:
        df = pd.read_parquet(filepath)
    except Exception as e:
        errors.append(f"Failed to read as Parquet: {e}")
        return checks, errors

    checks["row_count"] = len(df)
    checks["columns"] = list(df.columns)
    if len(df) == 0: errors.append("File opens but contains zero rows")

    if expected_columns:
        missing = [c for c in expected_columns if c not in df.columns]
        if missing: errors.append(f"Expected column(s) not found: {missing}")

    if date_column and date_column in df.columns:
        dates = pd.to_datetime(df[date_column])
        checks["date_range"] = f"{dates.min().date()} to {dates.max().date()}"

    return checks, errors

def check_shapefile(filepath: str, expected_geometry_types: Optional[List[str]] = None) -> Tuple[Dict[str, Any], List[str]]:
    import geopandas as gpd
    checks: Dict[str, Any] = {}
    errors: List[str] = []

    try:
        gdf = gpd.read_file(filepath)
    except Exception as e:
        errors.append(f"Failed to read as Shapefile: {e}")
        return checks, errors

    checks["feature_count"] = len(gdf)
    checks["geometry_types"] = list(gdf.geom_type.unique())
    if len(gdf) == 0: errors.append("File opens but contains zero features")

    return checks, errors

# ======================================================================
# Smart Duplicate Detection
# ======================================================================

def compute_checksum(filepath: str, chunk_size: int = 8192) -> str:
    h = hashlib.md5()
    with open(filepath, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk: break
            h.update(chunk)
    return h.hexdigest()

def find_true_duplicates(filepaths: List[str], max_size_mb_for_checksum: int = 500) -> Dict[str, List[str]]:
    """
    Groups files by checksum. Skips files > 500MB to save CPU 
    (massive NetCDF files are rarely accidentally duplicated).
    """
    checksum_map: Dict[str, List[str]] = {}
    for fp in filepaths:
        size_mb = os.path.getsize(fp) / (1024 * 1024)
        if size_mb > max_size_mb_for_checksum:
            continue # Skip huge files
            
        try:
            cs = compute_checksum(fp)
            checksum_map.setdefault(cs, []).append(fp)
        except Exception as e:
            logger.error(f"checksum_failed file={fp} error={e}")
            
    return {cs: files for cs, files in checksum_map.items() if len(files) > 1}

# ======================================================================
# THE PRODUCTION AGENT
# ======================================================================

class DataQualityAgent:
    def __init__(self, source_config: Optional[Dict[str, Dict[str, Any]]] = None, max_workers: int = 4):
        self.source_config = source_config or {}
        self.max_workers = max_workers # Number of parallel threads

    def _get_config_for(self, filepath: str) -> Dict[str, Any]:
        import fnmatch
        filename = os.path.basename(filepath)
        for pattern, cfg in self.source_config.items():
            if fnmatch.fnmatch(filename, pattern):
                return cfg
        return {}

    def check_file(self, filepath: str) -> FileCheckResult:
        cfg = self._get_config_for(filepath)
        ext = os.path.splitext(filepath)[1].lower()

        file_type = cfg.get("type", {".nc": "netcdf", ".csv": "csv", ".parquet": "parquet", ".shp": "shapefile"}.get(ext, "unknown"))

        if file_type == "netcdf":
            checks, errors = check_netcdf(filepath, cfg.get("required_date_start"), cfg.get("required_date_end"), cfg.get("expected_variables"))
        elif file_type == "csv":
            checks, errors = check_csv(filepath, cfg.get("expected_columns"), cfg.get("content_signature_key"))
        elif file_type == "parquet":
            checks, errors = check_parquet(filepath, cfg.get("expected_columns"), cfg.get("date_column"))
        elif file_type == "shapefile":
            checks, errors = check_shapefile(filepath, cfg.get("expected_geometry_types"))
        else:
            checks, errors = {}, [f"Unrecognized file type for extension '{ext}'"]

        return FileCheckResult(filepath=filepath, file_type=file_type, passed=(len(errors) == 0), checks=checks, errors=errors)

    def run_audit(self, filepaths: List[str], quarantine_dir: Optional[str] = None, supabase_client=None) -> AuditReport:
        report = AuditReport(run_timestamp=datetime.now(timezone.utc).isoformat())
        
        # 1. Multi-threaded file checking
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_path = {executor.submit(self.check_file, fp): fp for fp in filepaths}
            for future in as_completed(future_to_path):
                try:
                    result = future.result()
                    report.results.append(result)
                    logger.info(f"check_complete file={result.filepath} passed={result.passed}")
                except Exception as e:
                    fp = future_to_path[future]
                    logger.error(f"check_crashed file={fp} error={e}")
                    report.results.append(FileCheckResult(filepath=fp, file_type="unknown", passed=False, errors=[f"Crashed: {e}"]))

        # 2. Duplicate detection
        dupe_groups = find_true_duplicates(filepaths)
        for checksum, files in dupe_groups.items():
            for fp in files[1:]:
                for r in report.results:
                    if r.filepath == fp:
                        r.needs_human_review = True
                        r.review_reason = f"Byte-identical duplicate of {files[0]}"

        # 3. Quarantine failed files
        if quarantine_dir:
            os.makedirs(quarantine_dir, exist_ok=True)
            for r in report.failed_files():
                try:
                    dest = os.path.join(quarantine_dir, os.path.basename(r.filepath))
                    shutil.move(r.filepath, dest)
                    r.filepath = dest # Update path in report
                    logger.warning(f"quarantined file={dest}")
                except Exception as e:
                    logger.error(f"quarantine_failed file={r.filepath} error={e}")

        # 4. Log to Supabase
        if supabase_client:
            try:
                logs = [r.to_dict() for r in report.results]
                # Assumes you have a 'data_audit_log' table in Supabase
                supabase_client.table("data_audit_log").insert({
                    "run_timestamp": report.run_timestamp,
                    "results": logs # Stored as JSONB
                }).execute()
                logger.info("audit_log_uploaded_to_supabase")
            except Exception as e:
                logger.error(f"supabase_log_failed error={e}")

        return report
