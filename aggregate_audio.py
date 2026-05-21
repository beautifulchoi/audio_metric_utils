#!/usr/bin/env python3
"""Aggregate segmented EgoCom inference audio into scenario-level WAV files.

Inference audio directories are expected to be named like:

    sample_0000_day_1__con_4__part1/pred_audio.wav

The sample number is interpreted as a manifest row index. Rows are grouped by
``scene_name`` + ``src_person`` + ``tgt_person`` by default; each output
directory represents one directed transfer pair such as
``day_1__con_4__part1__person_1__person_3``. Source, target, and prediction
audio are concatenated only from rows matching that exact transfer pair.
Duplicate clips are de-duplicated by actual clip identity.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Iterable


DEFAULT_MANIFEST = Path(
    "/home/prj/data/egocom_holdout/4s_overlap0_v2_day1_con4_parts/test/manifest/manifest_mm.jsonl"
)
DEFAULT_AUDIO_ROOT = Path("/home/prj/ego2ego_mag/result/inference-geo-label/audio")
DEFAULT_OUTPUT_ROOT = Path(
    "/home/prj/ego2ego_mag/result/inference-geo-label/aggregated_audio"
)

SAMPLE_DIR_RE = re.compile(r"^sample_(\d+)_")
AUDIO_NAMES = ("pred_audio.wav", "src_audio.wav", "tgt_audio.wav")
DEFAULT_DEDUPE_AUDIO_NAMES = ("pred_audio.wav", "src_audio.wav", "tgt_audio.wav")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Concatenate per-sample pred/src/tgt WAV files into longer WAVs. "
            "By default groups are scene_name + src_person + tgt_person, "
            "producing one directory per directed transfer pair."
        )
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--audio-root", type=Path, default=DEFAULT_AUDIO_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--group-by",
        choices=("scene-person", "scene", "scene-src", "scene-src-tgt"),
        default="scene-src-tgt",
        help=(
            "Grouping key. Default scene-src-tgt keeps directed source-target "
            "transfer pairs separate, e.g. scene__person_1__person_3. "
            "scene-person is the legacy per-person timeline grouping. scene "
            "makes one whole-scenario group."
        ),
    )
    parser.add_argument(
        "--dedupe-audio-names",
        default=",".join(DEFAULT_DEDUPE_AUDIO_NAMES),
        help=(
            "Comma-separated audio names to de-duplicate before concatenation. "
            "Default: pred_audio.wav,src_audio.wav,tgt_audio.wav. Use '' or 'none' to disable."
        ),
    )
    parser.add_argument(
        "--sample-index-base",
        type=int,
        choices=(0, 1),
        default=0,
        help="Whether sample_0000 maps to manifest row 0 or row 1.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing aggregated WAV files.",
    )
    parser.add_argument(
        "--clean-output-root",
        action="store_true",
        help=(
            "Remove existing child directories under output-root before writing. "
            "Use this when changing grouping modes so stale aggregate groups are "
            "not included by downstream metric scripts."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned groups and outputs without running ffmpeg.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero on missing samples, missing audio files, or ffmpeg failures.",
    )
    parser.add_argument("--ffmpeg", default="ffmpeg", help="ffmpeg executable path.")
    return parser.parse_args()


def load_manifest(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
    return rows


def iter_sample_dirs(audio_root: Path) -> Iterable[tuple[int, Path]]:
    for sample_dir in sorted(audio_root.iterdir()):
        if not sample_dir.is_dir():
            continue
        match = SAMPLE_DIR_RE.match(sample_dir.name)
        if not match:
            continue
        yield int(match.group(1)), sample_dir


def build_sample_map(audio_root: Path, sample_index_base: int) -> dict[int, Path]:
    sample_map: dict[int, Path] = {}
    for sample_index, sample_dir in iter_sample_dirs(audio_root):
        manifest_index = sample_index - sample_index_base
        if manifest_index in sample_map:
            raise ValueError(
                f"Duplicate sample index {manifest_index}: "
                f"{sample_map[manifest_index]} and {sample_dir}"
            )
        sample_map[manifest_index] = sample_dir
    return sample_map


def group_manifest_rows(
    rows: list[dict],
    sample_map: dict[int, Path],
    group_by: str,
) -> dict[tuple[str, ...], list[dict]]:
    groups: dict[tuple[str, ...], list[dict]] = defaultdict(list)
    for row_index, row in enumerate(rows):
        sample_dir = sample_map.get(row_index)
        if sample_dir is None:
            continue
        scene_name = str(row.get("scene_name") or "")
        src_person = str(row.get("src_person") or "")
        if not scene_name or not src_person:
            raise ValueError(f"Manifest row {row_index} is missing scene_name or src_person")
        if group_by == "scene":
            group_key = (scene_name,)
        elif group_by == "scene-src-tgt":
            tgt_person = str(row.get("tgt_person") or "")
            if not tgt_person:
                raise ValueError(f"Manifest row {row_index} is missing tgt_person")
            group_key = (scene_name, src_person, tgt_person)
        else:
            group_key = (scene_name, src_person)
        group_row = dict(row)
        group_row["_row_index"] = row_index
        group_row["_sample_dir"] = sample_dir
        groups[group_key].append(group_row)

    for group_rows in groups.values():
        group_rows.sort(
            key=lambda row: (
                int(row.get("clip_start_ms", 0)),
                int(row.get("clip_end_ms", 0)),
                int(row["_row_index"]),
            )
        )
    return dict(groups)


def group_manifest_person_rows(
    rows: list[dict],
    sample_map: dict[int, Path],
) -> dict[tuple[str, str], dict[str, list[dict]]]:
    groups: dict[tuple[str, str], dict[str, list[dict]]] = defaultdict(
        lambda: {"pred_audio.wav": [], "src_audio.wav": [], "tgt_audio.wav": []}
    )
    for row_index, row in enumerate(rows):
        sample_dir = sample_map.get(row_index)
        if sample_dir is None:
            continue
        scene_name = str(row.get("scene_name") or "")
        src_person = str(row.get("src_person") or "")
        tgt_person = str(row.get("tgt_person") or "")
        if not scene_name or not src_person or not tgt_person:
            raise ValueError(
                f"Manifest row {row_index} is missing scene_name, src_person, or tgt_person"
            )

        group_row = dict(row)
        group_row["_row_index"] = row_index
        group_row["_sample_dir"] = sample_dir
        groups[(scene_name, src_person)]["src_audio.wav"].append(group_row)
        groups[(scene_name, tgt_person)]["tgt_audio.wav"].append(group_row)
        groups[(scene_name, tgt_person)]["pred_audio.wav"].append(group_row)

    for audio_rows_by_name in groups.values():
        for audio_rows in audio_rows_by_name.values():
            audio_rows.sort(
                key=lambda row: (
                    int(row.get("clip_start_ms", 0)),
                    int(row.get("clip_end_ms", 0)),
                    int(row["_row_index"]),
                )
            )
    return dict(groups)


def parse_dedupe_audio_names(value: str) -> set[str]:
    value = value.strip()
    if not value or value.lower() in {"none", "false", "0"}:
        return set()
    return {part.strip() for part in value.split(",") if part.strip()}


def safe_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    value = value.strip("._")
    return value or "unknown"


def output_dir_for_group(output_root: Path, group_key: tuple[str, ...]) -> Path:
    return output_root / "__".join(safe_name(value) for value in group_key)


def ffmpeg_concat_escape(path: Path) -> str:
    return str(path).replace("'", r"'\''")


def write_concat_list(paths: list[Path], concat_list_path: Path) -> None:
    with concat_list_path.open("w", encoding="utf-8") as handle:
        for path in paths:
            handle.write(f"file '{ffmpeg_concat_escape(path.resolve())}'\n")


def validate_group_audio(group_rows: list[dict]) -> tuple[bool, dict[str, list[str]]]:
    missing: dict[str, list[str]] = {}
    for audio_name in AUDIO_NAMES:
        missing_for_audio = [
            row["_sample_dir"].name
            for row in group_rows
            if not (row["_sample_dir"] / audio_name).is_file()
        ]
        if missing_for_audio:
            missing[audio_name] = missing_for_audio
    return not missing, missing


def dedupe_key_for_audio(row: dict, audio_name: str) -> tuple:
    if audio_name == "src_audio.wav":
        return (
            row.get("scene_name"),
            row.get("src_video_name"),
            row.get("src_clip_filename"),
            row.get("clip_start_ms"),
            row.get("clip_end_ms"),
        )
    if audio_name in {"tgt_audio.wav", "pred_audio.wav"}:
        return (
            row.get("scene_name"),
            row.get("tgt_video_name"),
            row.get("tgt_clip_filename"),
            row.get("clip_start_ms"),
            row.get("clip_end_ms"),
        )
    return (row["_row_index"],)


def rows_for_audio(group_rows: list[dict], audio_name: str, dedupe_audio_names: set[str]) -> list[dict]:
    if audio_name not in dedupe_audio_names:
        return group_rows

    seen = set()
    deduped_rows = []
    for row in group_rows:
        key = dedupe_key_for_audio(row, audio_name)
        if key in seen:
            continue
        seen.add(key)
        deduped_rows.append(row)
    return deduped_rows


def duplicate_rows_for_audio(
    group_rows: list[dict],
    audio_name: str,
    dedupe_audio_names: set[str],
) -> list[dict]:
    if audio_name not in dedupe_audio_names:
        return []

    seen = set()
    duplicates = []
    for row in group_rows:
        key = dedupe_key_for_audio(row, audio_name)
        if key in seen:
            duplicates.append(row)
            continue
        seen.add(key)
    return duplicates


def run_ffmpeg_concat(
    *,
    ffmpeg: str,
    audio_paths: list[Path],
    output_path: Path,
    overwrite: bool,
) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="aggregate_audio_") as tmpdir:
        concat_list = Path(tmpdir) / "concat.txt"
        write_concat_list(audio_paths, concat_list)
        cmd = [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y" if overwrite else "-n",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_list),
            "-c",
            "copy",
            str(output_path),
        ]
        result = subprocess.run(cmd, check=False)
    return result.returncode


def summarize_group(
    group_key: tuple[str, ...],
    group_rows: list[dict],
    out_dir: Path,
    dedupe_audio_names: set[str],
) -> dict:
    starts = [int(row.get("clip_start_ms", 0)) for row in group_rows]
    ends = [int(row.get("clip_end_ms", 0)) for row in group_rows]
    summary = {
        "group_key": list(group_key),
        "output_dir": str(out_dir),
        "num_pair_rows": len(group_rows),
        "manifest_row_indices": [int(row["_row_index"]) for row in group_rows],
        "sample_dirs": [row["_sample_dir"].name for row in group_rows],
        "clip_start_ms": min(starts) if starts else None,
        "clip_end_ms": max(ends) if ends else None,
        "manifest_duration_ms": sum(max(0, end - start) for start, end in zip(starts, ends)),
        "audio_segment_counts": {
            audio_name: len(rows_for_audio(group_rows, audio_name, dedupe_audio_names))
            for audio_name in AUDIO_NAMES
        },
    }
    summary["scene_name"] = group_key[0]
    if len(group_key) > 1:
        summary["src_person"] = group_key[1]
    if len(group_key) > 2:
        summary["tgt_person"] = group_key[2]
    return summary


def summarize_person_group(
    group_key: tuple[str, str],
    audio_rows_by_name: dict[str, list[dict]],
    out_dir: Path,
    dedupe_audio_names: set[str],
) -> dict:
    all_rows = [
        row
        for audio_name in AUDIO_NAMES
        for row in audio_rows_by_name.get(audio_name, [])
    ]
    starts = [int(row.get("clip_start_ms", 0)) for row in all_rows]
    ends = [int(row.get("clip_end_ms", 0)) for row in all_rows]
    scene_name, person = group_key
    duplicate_counts = {
        audio_name: len(
            duplicate_rows_for_audio(
                audio_rows_by_name.get(audio_name, []),
                audio_name,
                dedupe_audio_names,
            )
        )
        for audio_name in AUDIO_NAMES
    }
    return {
        "group_key": list(group_key),
        "scene_name": scene_name,
        "person": person,
        "output_dir": str(out_dir),
        "clip_start_ms": min(starts) if starts else None,
        "clip_end_ms": max(ends) if ends else None,
        "audio_segment_counts": {
            audio_name: len(
                rows_for_audio(
                    audio_rows_by_name.get(audio_name, []),
                    audio_name,
                    dedupe_audio_names,
                )
            )
            for audio_name in AUDIO_NAMES
        },
        "duplicate_segment_counts": duplicate_counts,
        "duplicate_manifest_row_indices": {
            audio_name: [
                int(row["_row_index"])
                for row in duplicate_rows_for_audio(
                    audio_rows_by_name.get(audio_name, []),
                    audio_name,
                    dedupe_audio_names,
                )
            ]
            for audio_name in AUDIO_NAMES
            if duplicate_counts[audio_name]
        },
    }


def clean_output_root(output_root: Path, dry_run: bool) -> list[str]:
    if not output_root.exists():
        return []

    removed_dirs = []
    for child in sorted(output_root.iterdir()):
        if not child.is_dir():
            continue
        removed_dirs.append(str(child))
        if not dry_run:
            shutil.rmtree(child)

    if removed_dirs:
        action = "DRY-RUN would remove" if dry_run else "removed"
        print(f"{action} {len(removed_dirs)} existing output dirs from {output_root}")
    return removed_dirs


def write_summary(output_root: Path, summary: dict, dry_run: bool) -> None:
    if dry_run:
        print(json.dumps(summary, indent=2))
        return
    output_root.mkdir(parents=True, exist_ok=True)
    summary_path = output_root / "summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    print(f"summary: {summary_path}")


def main() -> int:
    args = parse_args()

    if not args.manifest.is_file():
        print(f"ERROR missing manifest: {args.manifest}", file=sys.stderr)
        return 2
    if not args.audio_root.is_dir():
        print(f"ERROR missing audio root: {args.audio_root}", file=sys.stderr)
        return 2
    if shutil.which(args.ffmpeg) is None and not Path(args.ffmpeg).is_file():
        print(f"ERROR ffmpeg not found: {args.ffmpeg}", file=sys.stderr)
        return 2

    dedupe_audio_names = parse_dedupe_audio_names(args.dedupe_audio_names)
    rows = load_manifest(args.manifest)
    sample_map = build_sample_map(args.audio_root, args.sample_index_base)
    if args.group_by == "scene-person":
        person_groups = group_manifest_person_rows(rows, sample_map)
        groups = {}
    else:
        person_groups = {}
        groups = group_manifest_rows(rows, sample_map, args.group_by)
    missing_manifest_indices = [
        index for index in range(len(rows)) if index not in sample_map
    ]
    out_of_range_samples = [
        index for index in sorted(sample_map) if index < 0 or index >= len(rows)
    ]

    if missing_manifest_indices:
        print(
            f"WARNING {len(missing_manifest_indices)} manifest rows have no matching sample dir",
            file=sys.stderr,
        )
    if out_of_range_samples:
        print(
            f"WARNING {len(out_of_range_samples)} sample dirs do not map into the manifest",
            file=sys.stderr,
        )
    if args.strict and (missing_manifest_indices or out_of_range_samples):
        return 1

    print(f"manifest rows: {len(rows)}")
    print(f"sample dirs: {len(sample_map)}")
    num_groups = len(person_groups) if args.group_by == "scene-person" else len(groups)
    print(f"groups: {num_groups}")

    summary = {
        "manifest": str(args.manifest),
        "audio_root": str(args.audio_root),
        "output_root": str(args.output_root),
        "sample_index_base": args.sample_index_base,
        "group_by": args.group_by,
        "dedupe_audio_names": sorted(dedupe_audio_names),
        "clean_output_root": args.clean_output_root,
        "dry_run": args.dry_run,
        "strict": args.strict,
        "timing_policy": "strict_concat",
        "num_manifest_rows": len(rows),
        "num_sample_dirs": len(sample_map),
        "num_groups": num_groups,
        "missing_manifest_indices": missing_manifest_indices,
        "out_of_range_samples": out_of_range_samples,
        "cleaned_output_dirs": [],
        "groups": [],
        "skipped_groups": [],
        "errors": [],
    }

    if args.clean_output_root:
        summary["cleaned_output_dirs"] = clean_output_root(args.output_root, args.dry_run)

    had_error = False
    if args.group_by == "scene-person":
        for group_key in sorted(person_groups):
            audio_rows_by_name = person_groups[group_key]
            group_label = "/".join(group_key)
            out_dir = output_dir_for_group(args.output_root, group_key)
            group_summary = summarize_person_group(
                group_key,
                audio_rows_by_name,
                out_dir,
                dedupe_audio_names,
            )
            missing: dict[str, list[str]] = {}
            for audio_name in AUDIO_NAMES:
                audio_rows = rows_for_audio(
                    audio_rows_by_name.get(audio_name, []),
                    audio_name,
                    dedupe_audio_names,
                )
                missing_for_audio = [
                    row["_sample_dir"].name
                    for row in audio_rows
                    if not (row["_sample_dir"] / audio_name).is_file()
                ]
                if missing_for_audio:
                    missing[audio_name] = missing_for_audio

            if missing:
                message = (
                    f"missing audio for {group_label}: "
                    + ", ".join(f"{name}={len(samples)}" for name, samples in missing.items())
                )
                print(f"WARNING {message}", file=sys.stderr)
                group_summary["status"] = "skipped_missing_audio"
                group_summary["missing_audio"] = missing
                summary["skipped_groups"].append(group_summary)
                if args.strict:
                    had_error = True
                continue

            print(f"{group_label}: {group_summary['audio_segment_counts']} -> {out_dir}")
            group_summary["outputs"] = {}
            group_had_error = False

            for audio_name in AUDIO_NAMES:
                output_path = out_dir / audio_name
                audio_rows = rows_for_audio(
                    audio_rows_by_name.get(audio_name, []),
                    audio_name,
                    dedupe_audio_names,
                )
                audio_paths = [row["_sample_dir"] / audio_name for row in audio_rows]
                group_summary["outputs"][audio_name] = str(output_path)
                if output_path.exists() and not args.overwrite:
                    print(f"SKIP exists: {output_path}", file=sys.stderr)
                    continue
                if args.dry_run:
                    print(f"DRY-RUN {audio_name}: {len(audio_paths)} files -> {output_path}")
                    continue

                returncode = run_ffmpeg_concat(
                    ffmpeg=args.ffmpeg,
                    audio_paths=audio_paths,
                    output_path=output_path,
                    overwrite=args.overwrite,
                )
                if returncode != 0:
                    message = f"ffmpeg failed for {output_path}"
                    print(f"ERROR {message}", file=sys.stderr)
                    summary["errors"].append(message)
                    group_had_error = True
                    had_error = True
                    if args.strict:
                        break

            group_summary["status"] = "error" if group_had_error else "ok"
            summary["groups"].append(group_summary)
            if args.strict and group_had_error:
                break

        write_summary(args.output_root, summary, args.dry_run)
        return 1 if had_error else 0

    for group_key in sorted(groups):
        group_rows = groups[group_key]
        group_label = "/".join(group_key)
        out_dir = output_dir_for_group(args.output_root, group_key)
        group_summary = summarize_group(group_key, group_rows, out_dir, dedupe_audio_names)
        is_complete, missing = validate_group_audio(group_rows)
        if not is_complete:
            message = (
                f"missing audio for {group_label}: "
                + ", ".join(f"{name}={len(samples)}" for name, samples in missing.items())
            )
            print(f"WARNING {message}", file=sys.stderr)
            group_summary["status"] = "skipped_missing_audio"
            group_summary["missing_audio"] = missing
            summary["skipped_groups"].append(group_summary)
            if args.strict:
                had_error = True
            continue

        print(
            f"{group_label}: {len(group_rows)} segments -> {out_dir}"
        )
        group_summary["outputs"] = {}
        group_had_error = False

        for audio_name in AUDIO_NAMES:
            output_path = out_dir / audio_name
            audio_rows = rows_for_audio(group_rows, audio_name, dedupe_audio_names)
            audio_paths = [row["_sample_dir"] / audio_name for row in audio_rows]
            group_summary["outputs"][audio_name] = str(output_path)
            if output_path.exists() and not args.overwrite:
                print(f"SKIP exists: {output_path}", file=sys.stderr)
                continue
            if args.dry_run:
                print(f"DRY-RUN {audio_name}: {len(audio_paths)} files -> {output_path}")
                continue

            returncode = run_ffmpeg_concat(
                ffmpeg=args.ffmpeg,
                audio_paths=audio_paths,
                output_path=output_path,
                overwrite=args.overwrite,
            )
            if returncode != 0:
                message = f"ffmpeg failed for {output_path}"
                print(f"ERROR {message}", file=sys.stderr)
                summary["errors"].append(message)
                group_had_error = True
                had_error = True
                if args.strict:
                    break

        group_summary["status"] = "error" if group_had_error else "ok"
        summary["groups"].append(group_summary)
        if args.strict and group_had_error:
            break

    write_summary(args.output_root, summary, args.dry_run)
    return 1 if had_error else 0


if __name__ == "__main__":
    raise SystemExit(main())
