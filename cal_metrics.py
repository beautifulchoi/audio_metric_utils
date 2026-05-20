#!/usr/bin/env python3
"""Calculate audio metrics for segment-level or aggregated inference outputs."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchaudio
from pesq import pesq as pesq_score
from pesq import PesqError
from tqdm.auto import tqdm

SHARED_CODE_DIR = "/home/prj"
if SHARED_CODE_DIR not in sys.path:
    sys.path.append(SHARED_CODE_DIR)

from auraloss.freq import MultiResolutionSTFTLoss as MRSTFT

from comp_utils.metrics import AmplitudeLoss, L2Loss, PhaseLoss


MID_METRIC_NAMES = {"waveform_l2", "amplitude_l1", "phase"}
METADATA_COLUMNS = {"sample_rate", "num_samples", "duration_sec"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Calculate pred-vs-target audio metrics. By default this evaluates "
            "segment directories under audio/. Use --audio-mode aggregated for "
            "scenario/source-person outputs under aggregated_audio/."
        )
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/home/prj/ego-to-ego-audio-transfer/inference_result"),
        help="Path to inference result root.",
    )
    parser.add_argument(
        "--audio-mode",
        choices=("segments", "aggregated"),
        default="segments",
        help="Evaluate output_root/audio or output_root/aggregated_audio.",
    )
    parser.add_argument(
        "--audio-root",
        type=Path,
        default=None,
        help="Explicit directory containing audio item subdirectories. Overrides --audio-mode.",
    )
    parser.add_argument(
        "--output-prefix",
        default=None,
        help=(
            "CSV filename prefix. Defaults to 'metrics' for segments and "
            "'metrics_aggregated' for aggregated audio."
        ),
    )
    parser.add_argument(
        "--also-write-legacy",
        action="store_true",
        help="Also write metrics.csv and metrics_aggregate.csv.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Evaluate at most this many audio item directories.",
    )
    parser.add_argument(
        "--pesq-max-seconds",
        type=float,
        default=30.0,
        help=(
            "Maximum PESQ window length. Longer audio is split into windows "
            "and averaged. Use 0 to score the full audio in one PESQ call."
        ),
    )
    return parser.parse_args()


def resolve_audio_root(output_root: Path, audio_mode: str, audio_root: Path | None) -> Path:
    if audio_root is not None:
        return audio_root.expanduser().resolve()
    subdir = "aggregated_audio" if audio_mode == "aggregated" else "audio"
    return (output_root / subdir).resolve()


def default_output_prefix(audio_mode: str) -> str:
    return "metrics_aggregated" if audio_mode == "aggregated" else "metrics"


def build_metrics(sample_rate: int) -> dict[str, torch.nn.Module]:
    return {
        "waveform_l2": L2Loss(),
        "amplitude_l1": AmplitudeLoss(sample_rate=sample_rate),
        "phase": PhaseLoss(sample_rate=sample_rate),
        "mrstft": MRSTFT(),
        "pesq": None,
    }


def resample_if_needed(wav: torch.Tensor, orig_sr: int, target_sr: int) -> torch.Tensor:
    if orig_sr == target_sr:
        return wav
    return torchaudio.functional.resample(wav, orig_sr, target_sr)


def match_length(pred: torch.Tensor, gt: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    min_len = min(pred.shape[-1], gt.shape[-1])
    return pred[..., :min_len], gt[..., :min_len]


def should_skip_pesq(
    pred: torch.Tensor,
    gt: torch.Tensor,
    sr: int = 16000,
) -> tuple[bool, str]:
    min_len = int(0.25 * sr)

    if pred.shape[-1] < min_len or gt.shape[-1] < min_len:
        return True, "too_short"
    if not torch.isfinite(pred).all() or not torch.isfinite(gt).all():
        return True, "nan_or_inf"

    pred_rms = torch.sqrt(torch.mean(pred.float() ** 2)).item()
    gt_rms = torch.sqrt(torch.mean(gt.float() ** 2)).item()
    pred_peak = pred.abs().max().item()
    gt_peak = gt.abs().max().item()

    if gt_rms < 1e-4 or gt_peak < 1e-3:
        return True, f"silent_gt:rms={gt_rms:.2e},peak={gt_peak:.2e}"
    if pred_rms < 1e-5 or pred_peak < 1e-4:
        return True, f"silent_pred:rms={pred_rms:.2e},peak={pred_peak:.2e}"

    return False, "ok"


def iter_pesq_windows(
    pred: torch.Tensor,
    gt: torch.Tensor,
    sr: int,
    max_seconds: float,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    pred, gt = match_length(pred, gt)
    if max_seconds <= 0:
        return [(pred, gt)]

    window_size = int(max_seconds * sr)
    min_len = int(0.25 * sr)
    if window_size <= min_len:
        raise ValueError("--pesq-max-seconds must be greater than 0.25 seconds")

    windows = []
    for start in range(0, pred.shape[-1], window_size):
        end = min(start + window_size, pred.shape[-1])
        if end - start < min_len:
            continue
        windows.append((pred[..., start:end], gt[..., start:end]))
    return windows


def compute_ludlows_pesq(
    pred: torch.Tensor,
    gt: torch.Tensor,
    sample_rate: int,
    pesq_max_seconds: float,
) -> float:
    pred = resample_if_needed(pred, sample_rate, 16000).float()
    gt = resample_if_needed(gt, sample_rate, 16000).float()
    pred, gt = match_length(pred, gt)

    weighted_scores = []
    for pred_window, gt_window in iter_pesq_windows(pred, gt, 16000, pesq_max_seconds):
        skip, reason = should_skip_pesq(pred_window, gt_window, sr=16000)
        if skip:
            print(f"[SKIP PESQ WINDOW] {reason}")
            continue

        pred_np = pred_window.detach().cpu().numpy()
        gt_np = gt_window.detach().cpu().numpy()
        channels = min(pred_np.shape[0], gt_np.shape[0])
        window_samples = int(pred_window.shape[-1])
        for channel in range(channels):
            try:
                score = pesq_score(
                    16000,
                    gt_np[channel],
                    pred_np[channel],
                    "wb",
                )
            except PesqError as exc:
                print(f"[SKIP PESQ CHANNEL] {exc}")
                continue
            weighted_scores.append((float(score), window_samples))

    if not weighted_scores:
        raise RuntimeError("skip_pesq:no_valid_windows")

    total_samples = sum(samples for _, samples in weighted_scores)
    return float(sum(score * samples for score, samples in weighted_scores) / total_samples)


def iter_audio_dirs(audio_root: Path, limit: int | None) -> list[Path]:
    dirs = [path for path in sorted(audio_root.glob("*")) if path.is_dir()]
    if limit is not None:
        dirs = dirs[:limit]
    return dirs


def load_pair(item_dir: Path) -> tuple[torch.Tensor, torch.Tensor, int] | None:
    pred_path = item_dir / "pred_audio.wav"
    tgt_path = item_dir / "tgt_audio.wav"
    if not pred_path.exists() or not tgt_path.exists():
        print(f"[SKIP] missing wav: {item_dir.name}", file=sys.stderr)
        return None

    pred, pred_sr = torchaudio.load(pred_path)
    gt, gt_sr = torchaudio.load(tgt_path)
    if pred_sr != gt_sr:
        gt = resample_if_needed(gt, gt_sr, pred_sr)
    pred, gt = match_length(pred, gt)
    return pred, gt, pred_sr


def tensor_to_float(score) -> float:
    if isinstance(score, torch.Tensor):
        score = score.detach().cpu().item()
    return float(score)


def compute_metric(
    name: str,
    metric: torch.nn.Module,
    pred: torch.Tensor,
    gt: torch.Tensor,
    sample_rate: int,
    pesq_max_seconds: float,
) -> float:
    if name == "pesq":
        return compute_ludlows_pesq(pred, gt, sample_rate, pesq_max_seconds)
    elif name == "mrstft":
        pred_for_metric = pred.unsqueeze(0)
        gt_for_metric = gt.unsqueeze(0)
    else:
        pred_for_metric = pred
        gt_for_metric = gt

    return tensor_to_float(metric(pred_for_metric, gt_for_metric))


def compute_rows(audio_root: Path, limit: int | None, pesq_max_seconds: float) -> list[dict]:
    rows: list[dict] = []

    for item_dir in tqdm(iter_audio_dirs(audio_root, limit)):
        loaded = load_pair(item_dir)
        if loaded is None:
            continue
        pred, gt, sample_rate = loaded
        pred_mid = pred.mean(dim=0, keepdim=True)
        gt_mid = gt.mean(dim=0, keepdim=True)
        metrics = build_metrics(sample_rate)

        row: dict[str, float | int | str | None] = {
            "scene": item_dir.name,
            "sample_rate": sample_rate,
            "num_samples": int(pred.shape[-1]),
            "duration_sec": float(pred.shape[-1]) / float(sample_rate),
        }

        for name, metric in metrics.items():
            try:
                with torch.no_grad():
                    row[name] = compute_metric(
                        name,
                        metric,
                        pred,
                        gt,
                        sample_rate,
                        pesq_max_seconds,
                    )
            except RuntimeError as exc:
                if name == "pesq" and str(exc).startswith("skip_pesq:"):
                    print(f"[SKIP PESQ] {item_dir.name}, {str(exc).split(':', 1)[1]}")
                    row[name] = float("nan")
                else:
                    print(f"[WARN] {item_dir.name} / {name} failed: {exc}", file=sys.stderr)
                    row[name] = None
            except Exception as exc:
                print(f"[WARN] {item_dir.name} / {name} failed: {exc}", file=sys.stderr)
                row[name] = None

            if name in MID_METRIC_NAMES:
                mid_col_name = f"{name}_mid"
                try:
                    with torch.no_grad():
                        row[mid_col_name] = compute_metric(
                            name,
                            metric,
                            pred_mid,
                            gt_mid,
                            sample_rate,
                            pesq_max_seconds,
                        )
                except Exception as exc:
                    print(f"[WARN] {item_dir.name} / {mid_col_name} failed: {exc}", file=sys.stderr)
                    row[mid_col_name] = None

        rows.append(row)

    return rows


def save_outputs(
    output_root: Path,
    rows: list[dict],
    output_prefix: str,
    also_write_legacy: bool,
) -> tuple[Path, Path]:
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No metric rows were produced")

    df = df.set_index("scene")
    for col in ("waveform_l2", "waveform_l2_mid"):
        if col in df.columns:
            df[col] *= 1e5

    metrics_csv = output_root / f"{output_prefix}.csv"
    aggregate_csv = output_root / f"{output_prefix}_aggregate.csv"
    df.to_csv(metrics_csv)

    metric_cols = [col for col in df.columns if col not in METADATA_COLUMNS]
    agg = {}
    total_samples = int(pd.to_numeric(df["num_samples"], errors="coerce").fillna(0).sum())
    for col in metric_cols:
        mean, std, valid_count, valid_samples = weighted_mean_std(df[col], df["num_samples"])
        agg[col] = {
            "mean": mean,
            "std": std,
            "valid_count": valid_count,
            "total_count": len(df),
            "valid_samples": valid_samples,
            "total_samples": total_samples,
        }

    agg_df = pd.DataFrame(agg).T
    agg_df.to_csv(aggregate_csv, float_format="%.10f")

    if also_write_legacy:
        shutil.copy2(metrics_csv, output_root / "metrics.csv")
        shutil.copy2(aggregate_csv, output_root / "metrics_aggregate.csv")

    return metrics_csv, aggregate_csv


def weighted_mean_std(values: pd.Series, weights: pd.Series) -> tuple[float, float, int, int]:
    numeric_values = pd.to_numeric(values, errors="coerce")
    numeric_weights = pd.to_numeric(weights, errors="coerce")
    valid = numeric_values.notna() & numeric_weights.notna() & (numeric_weights > 0)
    if not valid.any():
        return float("nan"), float("nan"), 0, 0

    valid_values = numeric_values[valid].to_numpy(dtype=float)
    valid_weights = numeric_weights[valid].to_numpy(dtype=float)
    total_weight = int(valid_weights.sum())
    mean = float(np.average(valid_values, weights=valid_weights))
    variance = float(np.average((valid_values - mean) ** 2, weights=valid_weights))
    return mean, float(np.sqrt(variance)), int(valid.sum()), total_weight


def main() -> int:
    args = parse_args()
    output_root = args.output_root.expanduser().resolve()
    audio_root = resolve_audio_root(output_root, args.audio_mode, args.audio_root)
    output_prefix = args.output_prefix or default_output_prefix(args.audio_mode)

    if not output_root.exists():
        raise FileNotFoundError(f"output_root does not exist: {output_root}")
    if not audio_root.exists():
        raise FileNotFoundError(f"audio directory not found: {audio_root}")

    print(f"output_root: {output_root}")
    print(f"audio_root: {audio_root}")
    print(f"audio_mode: {args.audio_mode}")
    print(f"output_prefix: {output_prefix}")

    rows = compute_rows(audio_root, args.limit, args.pesq_max_seconds)
    metrics_csv, aggregate_csv = save_outputs(
        output_root,
        rows,
        output_prefix,
        args.also_write_legacy,
    )

    print(f"Saved metrics to {metrics_csv}")
    print(f"Saved aggregate csv to {aggregate_csv}")
    if args.also_write_legacy:
        print(f"Also wrote legacy CSV names under {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
