#!/usr/bin/env bash
set -euo pipefail

# Edit this list, or pass output roots / metrics CSV files as CLI args:
#   /home/prj/aggregate_result.sh /path/to/project_a/inference_result /path/to/project_b/metrics_aggregate.csv
OUTPUT_ROOTS=(
    /home/prj/ego2ego_mag/result/inference-geo-label/metrics_aggregated_aggregate.csv
    /home/prj/ego2ego_mag/result/inference-text/metrics_aggregated_aggregate.csv
    /home/prj/ego2ego_mag/result/inference-vision/metrics_aggregated_aggregate.csv
    /home/prj/comparision_baselines_new/DAVIS/inference_result/egocom-vision-fm/metrics_aggregated_aggregate.csv


)

OUT_CSV="${OUT_CSV:-/home/prj/metrics_mean_aggregate.csv}"

if [[ "$#" -gt 0 ]]; then
  OUTPUT_ROOTS=("$@")
fi

if [[ "${#OUTPUT_ROOTS[@]}" -eq 0 ]]; then
  echo "No output roots provided. Edit OUTPUT_ROOTS or pass paths as arguments." >&2
  exit 2
fi

python - "${OUT_CSV}" "${OUTPUT_ROOTS[@]}" <<'PY'
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

out_csv = Path(sys.argv[1]).expanduser()
output_roots = [Path(arg).expanduser() for arg in sys.argv[2:]]

rows = []
metric_order = []
PROJECT_ROOT = Path("/home/prj")

def format_decimal(value):
    value = value.strip()
    if not value:
        return value
    try:
        return f"{float(value):.4f}"
    except ValueError:
        return value

def project_name_from_path(input_path):
    try:
        relative_path = input_path.resolve().relative_to(PROJECT_ROOT)
    except ValueError:
        return input_path.parent.name
    return relative_path.parts[0] if relative_path.parts else input_path.name

def project_name_from_csv(input_path):
    try:
        relative_path = input_path.resolve().relative_to(PROJECT_ROOT)
    except ValueError:
        return input_path.stem
    return relative_path.parts[0] if len(relative_path.parts) > 1 else input_path.stem

def resolve_metrics_path(input_path):
    if input_path.is_file():
        if input_path.suffix.lower() == ".csv":
            project_name = project_name_from_csv(input_path)
            return input_path, project_name, f"{project_name}/{input_path.stem}"
        print(f"Warning: not a CSV file, skipping: {input_path}", file=sys.stderr)
        return None, None, None

    metrics_path = input_path / "metrics_aggregate.csv"
    if metrics_path.is_file():
        project_name = project_name_from_path(input_path)
        return metrics_path, project_name, f"{project_name}/{input_path.name}"

    print(f"Warning: missing metrics_aggregate.csv, skipping: {metrics_path}", file=sys.stderr)
    return None, None, None

entries = []
for input_path in output_roots:
    metrics_path, project_name, detailed_project_name = resolve_metrics_path(input_path)
    if metrics_path is None:
        continue
    entries.append((metrics_path, project_name, detailed_project_name))

project_counts = Counter(project_name for _, project_name, _ in entries)
used_names = defaultdict(int)

for metrics_path, project_name, detailed_project_name in entries:
    if project_counts[project_name] > 1:
        project_name = detailed_project_name

    used_names[project_name] += 1
    if used_names[project_name] > 1:
        project_name = f"{project_name}#{used_names[project_name]}"

    values = {"project": project_name}

    with metrics_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        metric_field = reader.fieldnames[0] if reader.fieldnames else None
        if metric_field is None or "mean" not in reader.fieldnames:
            raise ValueError(f"Expected metric-name column and mean column in {metrics_path}")

        for row in reader:
            metric = (row.get(metric_field) or "").strip()
            mean = (row.get("mean") or "").strip()
            if not metric:
                continue
            if metric.endswith("_mid"):
                continue
            if metric not in metric_order:
                metric_order.append(metric)
            values[metric] = format_decimal(mean)

    rows.append(values)

out_csv.parent.mkdir(parents=True, exist_ok=True)
with out_csv.open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=["project", *metric_order])
    writer.writeheader()
    writer.writerows(rows)

print(out_csv)
PY
