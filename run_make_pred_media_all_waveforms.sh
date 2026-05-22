#!/usr/bin/env bash
set -uo pipefail

DATA_ROOT="/home/prj/data/egocom_holdout/4s_overlap0_v2_day1_con4_parts/test"
SCRIPT="/home/prj/comp_utils/make_pred_media.sh"
MANAGER_LOG="/home/prj/comp_utils/run_make_pred_media_all_waveforms.manager.log"

roots=(
  "/home/prj/ego2ego_mag/result/inference-geo-label"
  "/home/prj/ego2ego_mag/result/inference-text"
  "/home/prj/ego2ego_mag/result/inference-vision"
  "/home/prj/comparision_baselines_new/DAVIS/inference_result/egocom-vision-fm"
)

printf '[%s] manager start\n' "$(date -Is)" > "$MANAGER_LOG"

pids=()
for root in "${roots[@]}"; do
  log="$root/make_pred_media_all_waveforms.log"
  pidfile="$root/make_pred_media_all_waveforms.pid"
  statusfile="$root/make_pred_media_all_waveforms.status"
  : > "$log"
  printf '[%s] starting %s\n' "$(date -Is)" "$root" | tee "$statusfile" >> "$MANAGER_LOG"
  env PYTHONUNBUFFERED=1     INFER_ROOT="$root"     DATA_ROOT="$DATA_ROOT"     OUTPUT_ROOT="$root/pred_media"     "$SCRIPT" --overwrite >> "$log" 2>&1 &
  pid=$!
  echo "$pid" > "$pidfile"
  pids+=("$pid:$root")
  printf '[%s] pid=%s root=%s log=%s\n' "$(date -Is)" "$pid" "$root" "$log" >> "$MANAGER_LOG"
done

failed=0
for item in "${pids[@]}"; do
  pid="${item%%:*}"
  root="${item#*:}"
  statusfile="$root/make_pred_media_all_waveforms.status"
  if wait "$pid"; then
    printf '[%s] completed pid=%s root=%s\n' "$(date -Is)" "$pid" "$root" | tee "$statusfile" >> "$MANAGER_LOG"
  else
    rc=$?
    failed=1
    printf '[%s] failed rc=%s pid=%s root=%s\n' "$(date -Is)" "$rc" "$pid" "$root" | tee "$statusfile" >> "$MANAGER_LOG"
  fi
done

printf '[%s] manager done failed=%s\n' "$(date -Is)" "$failed" >> "$MANAGER_LOG"
exit "$failed"
