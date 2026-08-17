"""
BlueFish AI - Model Loader
============================
On startup, downloads all model artifacts from Supabase Storage to the
local tmp/ directory and loads them into a shared ModelRegistry in memory.

This is the single source of truth for loaded model instances.
All API routes and agents import from here.

Download order:
  1. ONNX files (model1)
  2. Pickle files (model6, model9, model11)
  3. PyTorch checkpoint (model3)
  4. Pure-python models (model2, model5, model7, model8, model10) — instantiated directly

Models 2, 5, 7, 8, 10 are pure math — no binary artifacts to download.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger("bluefish.model_loader")

# ── Add MODELS directory to path so model*.py scripts can be imported ─────────
# MODELS/ lives inside BLUEFISH_AI-BACKEND/, so it's two levels up from core/
MODELS_BASE_DIR = Path(__file__).parent.parent / "MODELS"
if MODELS_BASE_DIR.exists():
    for model_dir in MODELS_BASE_DIR.iterdir():
        if model_dir.is_dir():
            sys.path.insert(0, str(model_dir))
else:
    logger.warning(f"MODELS directory not found at {MODELS_BASE_DIR}. Model imports may fail.")

TMP_DIR = Path(__file__).parent.parent / "tmp"
TMP_DIR.mkdir(exist_ok=True)


class ModelRegistry:
    """
    Global in-memory registry for all 11 AI model instances.
    Populated once at startup. Read-only after that.
    """

    def __init__(self):
        # ONNX Runtime sessions (Model 1)
        self.model1: Optional[Any] = None          # Model1Client from AGENTS/agent_1.py

        # Pure-math models (instantiated directly, no artifacts)
        self.model2: Optional[Any] = None          # OceanFrontEddyModel
        self.model5: Optional[Any] = None          # FleetDensityModel
        self.model7: Optional[Any] = None          # RouteOptimizationModel
        self.model8: Optional[Any] = None          # TimeWindowModel
        self.model10: Optional[Any] = None         # CollisionDetectionModel

        # PyTorch LSTM (Model 3)
        self.model3: Optional[Any] = None          # Seq2SeqLSTMService

        # TFT (Model 4) — optional dependency
        self.model4: Optional[Any] = None          # TFTForecastService

        # Isolation Forest (Model 6)
        self.model6_forest: Optional[Any] = None
        self.model6_scaler: Optional[Any] = None

        # KMeans (Model 9)
        self.model9_kmeans: Optional[Any] = None

        # XGBoost Climate (Model 11)
        self.model11_xgb: Optional[Any] = None
        self.model11_scaler: Optional[Any] = None

        self._load_errors: Dict[str, str] = {}

    def record_error(self, model_name: str, error: str):
        self._load_errors[model_name] = error
        logger.error(f"MODEL LOAD FAILED: {model_name} — {error}")

    @property
    def load_errors(self) -> Dict[str, str]:
        return self._load_errors


# Singleton instance
_registry = ModelRegistry()


def get_model_registry() -> ModelRegistry:
    return _registry


def _download_from_supabase(
    supabase_client: Any,
    bucket: str,
    storage_path: str,
    local_path: Path,
) -> bool:
    """Downloads a single file from Supabase Storage. Returns True on success."""
    if local_path.exists():
        logger.info(f"Using cached artifact: {local_path.name}")
        return True
    try:
        logger.info(f"Downloading {storage_path} from bucket '{bucket}'...")
        data = supabase_client.storage.from_(bucket).download(storage_path)
        local_path.write_bytes(data)
        logger.info(f"Downloaded {local_path.name} ({local_path.stat().st_size / 1024:.1f} KB)")
        return True
    except Exception as e:
        logger.error(f"Failed to download {storage_path}: {e}")
        return False


async def load_all_models(supabase_client: Any, bucket: str = "ml-models") -> ModelRegistry:
    """
    Downloads all model artifacts and loads them into the ModelRegistry.
    Called once in FastAPI's startup event handler.
    """
    import pickle
    reg = get_model_registry()

    # ── Model 1: PFZ (ONNX) ───────────────────────────────────────────────────
    try:
        stage1_path = TMP_DIR / "stage1_presence.onnx"
        stage2_path = TMP_DIR / "stage2_intensity.onnx"
        ok1 = _download_from_supabase(supabase_client, bucket, "model1/stage1_presence.onnx", stage1_path)
        ok2 = _download_from_supabase(supabase_client, bucket, "model1/stage2_intensity.onnx", stage2_path)

        if ok1 and ok2:
            # Import Model1Client from the advisory agent
            from agents.advisory_agent import Model1Client
            from core.config import get_settings
            settings = get_settings()
            reg.model1 = Model1Client(
                str(stage1_path),
                str(stage2_path),
                feature_order=settings.MODEL1_FEATURE_ORDER,
            )
            logger.info("✓ Model 1 (PFZ ONNX) loaded.")
        else:
            reg.record_error("model1", "Failed to download ONNX artifacts from Supabase Storage.")
    except Exception as e:
        reg.record_error("model1", str(e))

    # ── Model 2: Fronts/Eddies (Pure Math) ───────────────────────────────────
    try:
        from model2 import OceanFrontEddyModel
        reg.model2 = OceanFrontEddyModel()
        logger.info("✓ Model 2 (Fronts/Eddies) instantiated.")
    except Exception as e:
        reg.record_error("model2", str(e))

    # ── Model 3: Migration LSTM (PyTorch) ─────────────────────────────────────
    try:
        pt_path = TMP_DIR / "model3_seq2seq_lstm.pt"
        ok = _download_from_supabase(supabase_client, bucket, "model3/model3_seq2seq_lstm.pt", pt_path)
        if ok:
            from services.model3_service import Seq2SeqLSTMService
            reg.model3 = Seq2SeqLSTMService(str(pt_path))
            logger.info("✓ Model 3 (Migration LSTM) loaded.")
        else:
            reg.record_error("model3", "Failed to download PyTorch checkpoint.")
    except Exception as e:
        reg.record_error("model3", str(e))

    # ── Model 4: TFT Seasonal (PyTorch Forecasting) ───────────────────────────
    try:
        ckpt_path = TMP_DIR / "tft_model.ckpt"
        ok = _download_from_supabase(supabase_client, bucket, "model4/tft_model.ckpt", ckpt_path)
        if ok:
            from services.model4_service import TFTForecastService
            reg.model4 = TFTForecastService(str(ckpt_path))
            logger.info("✓ Model 4 (TFT Seasonal) loaded.")
        else:
            reg.record_error("model4", "Failed to download TFT checkpoint.")
    except Exception as e:
        # TFT is optional — pytorch_forecasting may not be installed on all builds
        reg.record_error("model4", f"TFT load skipped (optional): {e}")

    # ── Model 5: Fleet Density (Pure DBSCAN, no artifact) ────────────────────
    try:
        from model5 import FleetDensityModel
        reg.model5 = FleetDensityModel()
        logger.info("✓ Model 5 (Fleet Density) instantiated.")
    except Exception as e:
        reg.record_error("model5", str(e))

    # ── Model 6: Anomaly Detection (Isolation Forest + Scaler) ───────────────
    try:
        iso_path = TMP_DIR / "isolation_forest.pkl"
        scaler6_path = TMP_DIR / "scaler_m6.pkl"
        ok1 = _download_from_supabase(supabase_client, bucket, "model6/isolation_forest.pkl", iso_path)
        ok2 = _download_from_supabase(supabase_client, bucket, "model6/scaler.pkl", scaler6_path)
        if ok1 and ok2:
            with open(iso_path, "rb") as f:
                reg.model6_forest = pickle.load(f)
            with open(scaler6_path, "rb") as f:
                reg.model6_scaler = pickle.load(f)
            logger.info("✓ Model 6 (Isolation Forest) loaded.")
        else:
            reg.record_error("model6", "Failed to download isolation_forest.pkl or scaler.pkl")
    except Exception as e:
        reg.record_error("model6", str(e))

    # ── Model 7: Route Optimization (Pure A*, no artifact) ───────────────────
    try:
        from model7 import RouteOptimizationModel
        reg.model7 = RouteOptimizationModel()
        logger.info("✓ Model 7 (Route Optimization) instantiated.")
    except Exception as e:
        reg.record_error("model7", str(e))

    # ── Model 8: Time Window (Pure Solunar Math, no artifact) ────────────────
    try:
        from model8 import TimeWindowModel
        reg.model8 = TimeWindowModel()
        logger.info("✓ Model 8 (Time Window) instantiated.")
    except Exception as e:
        reg.record_error("model8", str(e))

    # ── Model 9: Vessel Segmentation (KMeans) ────────────────────────────────
    try:
        km_path = TMP_DIR / "kmeans_model.pkl"
        ok = _download_from_supabase(supabase_client, bucket, "model9/kmeans_model.pkl", km_path)
        if ok:
            with open(km_path, "rb") as f:
                reg.model9_kmeans = pickle.load(f)
            logger.info("✓ Model 9 (KMeans Segmentation) loaded.")
        else:
            reg.record_error("model9", "Failed to download kmeans_model.pkl")
    except Exception as e:
        reg.record_error("model9", str(e))

    # ── Model 10: Collision Detection (Pure CPA/TCPA, no artifact) ───────────
    try:
        from model10 import CollisionDetectionModel
        from core.config import get_settings
        s = get_settings()
        reg.model10 = CollisionDetectionModel(
            cpa_threshold_km=s.COLLISION_CPA_THRESHOLD_KM,
            tcpa_threshold_min=s.COLLISION_TCPA_THRESHOLD_MIN,
        )
        logger.info("✓ Model 10 (Collision Detection) instantiated.")
    except Exception as e:
        reg.record_error("model10", str(e))

    # ── Model 11: Climate Risk (XGBoost + Scaler) ────────────────────────────
    try:
        xgb_path = TMP_DIR / "XGBoost_Boss_Model.pkl"
        scaler11_path = TMP_DIR / "scaler_Boss.pkl"
        ok1 = _download_from_supabase(supabase_client, bucket, "model11/XGBoost_Boss_Model.pkl", xgb_path)
        ok2 = _download_from_supabase(supabase_client, bucket, "model11/scaler_Boss.pkl", scaler11_path)
        if ok1 and ok2:
            with open(xgb_path, "rb") as f:
                reg.model11_xgb = pickle.load(f)
            with open(scaler11_path, "rb") as f:
                reg.model11_scaler = pickle.load(f)
            logger.info("✓ Model 11 (XGBoost Climate) loaded.")
        else:
            reg.record_error("model11", "Failed to download XGBoost model or scaler.")
    except Exception as e:
        reg.record_error("model11", str(e))

    # ── Summary ───────────────────────────────────────────────────────────────
    failed = list(reg.load_errors.keys())
    loaded = 11 - len(failed)
    logger.info(f"Model loading complete: {loaded}/11 loaded. Failures: {failed or 'none'}")

    return reg
