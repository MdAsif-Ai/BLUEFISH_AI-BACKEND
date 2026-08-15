#!/usr/bin/env bash
# =============================================================================
# run_model3_pipeline.sh
# Master runner for Model 3 â€” Short-Term Fishing Zone Migration Forecast
# =============================================================================
# Usage:  bash run_model3_pipeline.sh
# Runs all four steps in sequence, logging each to its own .log file.
# =============================================================================

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="$SCRIPT_DIR/.venv/bin/python"
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"

echo "======================================================"
echo "  BlueFish AI â€” Model 3 Pipeline"
echo "  Short-Term Effort Migration Forecast (Seq2Seq LSTM)"
echo "======================================================"
echo "  Working dir : $SCRIPT_DIR"
echo "  Python      : $PYTHON"
echo "  Start time  : $(date)"
echo "======================================================"

run_step() {
    local step_num="$1"
    local script="$2"
    local desc="$3"
    local log="$LOG_DIR/step${step_num}.log"

    echo ""
    echo "â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€"
    echo "  STEP $step_num: $desc"
    echo "â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€"
    echo "  Log: $log"
    echo ""

    "$PYTHON" "$SCRIPT_DIR/$script" 2>&1 | tee "$log"
    local exit_code=${PIPESTATUS[0]}

    if [ $exit_code -ne 0 ]; then
        echo ""
        echo "âŒ Step $step_num FAILED (exit code $exit_code)"
        echo "   Check log: $log"
        exit $exit_code
    fi

    echo ""
    echo "âœ… Step $step_num complete."
}

run_step 1 "step1_generate_synthetic_model1.py" \
           "Generate Synthetic Model-1 Daily Probability Maps"

run_step 2 "step2_build_sliding_windows.py" \
           "Build Sliding Window Sequences"

run_step 3 "step3_train_seq2seq_lstm.py" \
           "Train Seq2Seq LSTM (PyTorch)"

run_step 4 "step4_forecast_inference.py" \
           "Run Forecast Inference & Generate Maps"

echo ""
echo "======================================================"
echo "  ALL STEPS COMPLETE âœ…"
echo "  End time: $(date)"
echo ""
echo "  Output files:"
echo "    model1_daily_predictions.parquet"
echo "    model3_seq2seq_lstm.pt       â† trained model state dict"
echo "    model3_architecture.pt       â† full model for inference"
echo "    training_history.csv"
echo "    evaluation_report.txt"
echo "    forecast_3day.png"
echo "    forecast_7day.png"
echo "    forecast_14day.png"
echo "    forecast_comparison.png"
echo "======================================================"
