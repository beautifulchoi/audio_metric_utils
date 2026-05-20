#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MANIFEST="${MANIFEST:-/home/prj/data/egocom_holdout/4s_overlap0_v2_day1_con4_parts/test/manifest/manifest_mm.jsonl}"
AUDIO_ROOT="${AUDIO_ROOT:-/home/prj/ego2ego_mag/result/inference-geo-label/audio}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/prj/ego2ego_mag/result/inference-geo-label/aggregated_audio}"

exec python3 "${SCRIPT_DIR}/aggregate_audio.py" \
  --manifest "${MANIFEST}" \
  --audio-root "${AUDIO_ROOT}" \
  --output-root "${OUTPUT_ROOT}" \
  "$@"
