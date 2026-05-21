#!/usr/bin/env bash
set -euo pipefail

# Python executable (override if needed)
PYTHON_BIN="${PYTHON_BIN:-python}"
AUDIO_MODE="${AUDIO_MODE:-both}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-}"
LIMIT="${LIMIT:-}"
ALSO_WRITE_LEGACY="${ALSO_WRITE_LEGACY:-0}"
PESQ_MAX_SECONDS="${PESQ_MAX_SECONDS:-30}"

# Override this list at runtime with OUTPUT_ROOTS_OVERRIDE as a colon-separated
# list of output_root directories.
# Example:
# OUTPUT_ROOTS_OVERRIDE="/path/to/run_a:/path/to/run_b" ./run_cal_metrics_batch.sh
#
# Or update this list with output_root directories you want to evaluate.
# Example:
# OUTPUT_ROOTS=(
#   "/home/prj/ego-to-ego-audio-transfer/inference_result"
#   "/home/prj/ego-to-ego-audio-transfer_fmodulate/inference_result"
# )
if [[ -n "${OUTPUT_ROOTS_OVERRIDE:-}" ]]; then
  IFS=':' read -r -a OUTPUT_ROOTS <<< "${OUTPUT_ROOTS_OVERRIDE}"
else
  OUTPUT_ROOTS=(
    /home/prj/comparision_baselines_new/DAVIS/inference_result/egocom-vision-fm-10s
  )
fi

SCRIPT_PATH="/home/prj/comp_utils/cal_metrics.py"

if [[ "${AUDIO_MODE}" == "both" ]]; then
  AUDIO_MODES=(segments aggregated)
else
  AUDIO_MODES=("${AUDIO_MODE}")
fi

for output_root in "${OUTPUT_ROOTS[@]}"; do
  for mode in "${AUDIO_MODES[@]}"; do
  args=(
    "${SCRIPT_PATH}"
    --output-root "${output_root}"
    --audio-mode "${mode}"
    --pesq-max-seconds "${PESQ_MAX_SECONDS}"
  )

  if [[ -n "${OUTPUT_PREFIX}" ]]; then
    if [[ "${AUDIO_MODE}" == "both" ]]; then
      args+=(--output-prefix "${OUTPUT_PREFIX}_${mode}")
    else
      args+=(--output-prefix "${OUTPUT_PREFIX}")
    fi
  fi
  if [[ -n "${LIMIT}" ]]; then
    args+=(--limit "${LIMIT}")
  fi
  if [[ "${ALSO_WRITE_LEGACY}" == "1" || "${ALSO_WRITE_LEGACY}" == "true" ]]; then
    args+=(--also-write-legacy)
  fi

  echo "[RUN] output_root=${output_root} audio_mode=${mode}"
  "${PYTHON_BIN}" "${args[@]}"
  echo "[DONE] output_root=${output_root} audio_mode=${mode}"
  echo
  done

done

echo "All metric runs completed."
