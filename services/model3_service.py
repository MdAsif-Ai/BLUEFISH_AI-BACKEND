"""
BlueFish AI - Model 3 Service (Seq2Seq LSTM Migration)
=======================================================
Wraps the PyTorch LSTM model for fish migration trajectory forecasting.
The model takes a sequence of past observations and predicts future positions.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import torch

logger = logging.getLogger("bluefish.model3_service")


class Seq2SeqLSTMService:
    """
    Service wrapper for the Seq2Seq LSTM migration model.

    The model expects input tensors of shape (batch, seq_len, n_features)
    and outputs a predicted trajectory (sequence of lat/lon pairs).

    Since the model was trained on specific spatial features (SST, currents,
    chlorophyll, temporal encoding), this service provides a simplified
    inference path using the raw checkpoint.
    """

    def __init__(self, checkpoint_path: str, device: str = "cpu"):
        self.device = torch.device(device)
        logger.info(f"Loading LSTM checkpoint from {checkpoint_path}...")
        # Load the full checkpoint dict
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        # The checkpoint may be a raw state_dict or a full checkpoint with metadata
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            self.model = self._build_model_from_checkpoint(checkpoint)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.input_size = checkpoint.get("input_size", 10)
            self.output_steps = checkpoint.get("output_steps", 7)
        else:
            # It's a raw state dict or full model — handle gracefully
            logger.warning("Checkpoint format is non-standard. Storing raw for passthrough inference.")
            self.model = checkpoint  # May be a full nn.Module
            self.input_size = 10
            self.output_steps = 7

        if hasattr(self.model, "eval"):
            self.model.eval()
            self.model.to(self.device)
        logger.info("✓ LSTM model ready.")

    @staticmethod
    def _build_model_from_checkpoint(checkpoint: dict):
        """Reconstructs the LSTM architecture from checkpoint metadata."""
        import torch.nn as nn

        input_size = checkpoint.get("input_size", 10)
        hidden_size = checkpoint.get("hidden_size", 128)
        output_size = checkpoint.get("output_size", 2)  # lat, lon
        num_layers = checkpoint.get("num_layers", 2)

        class MigrationLSTM(nn.Module):
            def __init__(self):
                super().__init__()
                self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
                self.fc = nn.Linear(hidden_size, output_size)

            def forward(self, x):
                out, _ = self.lstm(x)
                return self.fc(out[:, -1, :])  # last timestep

        return MigrationLSTM()

    def predict(self, feature_sequence: List[List[float]], steps: int = 7) -> Dict[str, Any]:
        """
        Runs migration forecast.

        Args:
            feature_sequence: List of [n_features] rows (time steps), shape (seq_len, n_features)
            steps: Number of future time steps to forecast (default 7 days)

        Returns:
            Dict with 'trajectory' (list of {lat, lon}) and 'confidence'
        """
        try:
            x = torch.tensor([feature_sequence], dtype=torch.float32).to(self.device)
            with torch.no_grad():
                output = self.model(x)

            # Output shape: (1, 2) → [predicted_lat, predicted_lon]
            pred = output.squeeze().cpu().numpy()

            if pred.ndim == 1 and len(pred) == 2:
                trajectory = [{"lat": float(pred[0]), "lon": float(pred[1]), "day": 1}]
            else:
                trajectory = [
                    {"lat": float(pred[i, 0]), "lon": float(pred[i, 1]), "day": i + 1}
                    for i in range(min(steps, len(pred)))
                ]

            return {
                "trajectory": trajectory,
                "forecast_days": len(trajectory),
                "confidence": "medium",  # Placeholder — replace with ensemble variance if available
            }
        except Exception as e:
            logger.error(f"Model 3 inference failed: {e}")
            raise
