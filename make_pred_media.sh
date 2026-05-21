#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INFER_ROOT="${INFER_ROOT:-/home/prj/ego2ego_complex/result/egocom-metadata-geodata-src/visualization/validation_step_00040000}"
DATA_ROOT="${DATA_ROOT:-/home/prj/data/egocom_holdout/4s_overlap0_v2_day1_con4_parts/test}"
AUDIO_FLIPPED_ARGS="${AUDIO_FLIPPED_ARGS---audio-flipped}"

exec python "${SCRIPT_DIR}/make_pred_media.py" \
  --infer-root "${INFER_ROOT}" \
  --summary "${SUMMARY:-${INFER_ROOT}/summary.json}" \
  --data-root "${DATA_ROOT}" \
  --video-root "${VIDEO_ROOT:-${DATA_ROOT}/video}" \
  --audio-root "${AUDIO_ROOT:-${INFER_ROOT}/audio}" \
  --plot-root "${PLOT_ROOT:-${INFER_ROOT}/plots}" \
  --output-root "${OUTPUT_ROOT:-${INFER_ROOT}/pred_media}" \
  --media-kind "${MEDIA_KIND:-all}" \
  ${AUDIO_FLIPPED_ARGS} \
  "$@"
