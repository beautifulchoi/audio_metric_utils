#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INFER_ROOT="${INFER_ROOT:-/home/prj/ego2ego_only_mag/inference_result/geocue_True}"
AUDIO_FLIPPED_ARGS="${AUDIO_FLIPPED_ARGS---audio-flipped}"

exec python "${SCRIPT_DIR}/make_pred_media.py" \
  --infer-root "${INFER_ROOT}" \
  --video-root "${VIDEO_ROOT:-/home/prj/data/egocom/480p/5min_parts_10s_stride5_exact10}" \
  --audio-root "${AUDIO_ROOT:-${INFER_ROOT}/audio}" \
  --plot-root "${PLOT_ROOT:-${INFER_ROOT}/plots}" \
  --manifest "${MANIFEST:-/home/prj/ego2ego_only_mag/manifests/egocom_test_pairs_geocue_target_exact10.jsonl}" \
  --output-root "${OUTPUT_ROOT:-${INFER_ROOT}/pred_media}" \
  --media-kind "${MEDIA_KIND:-all}" \
  ${AUDIO_FLIPPED_ARGS} \
  "$@"
