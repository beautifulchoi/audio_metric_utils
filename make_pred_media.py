#!/usr/bin/env python3
"""Create EgoCom source/target/prediction videos with animated audio plots.

Predicted audio directories are expected to be named like:

    sample_0155_day_1__con_4__part1/pred_audio.wav

The sample number is interpreted as a 0-based row index into the manifest
JSONL. For each selected sample this script writes source, target, and/or pred
media by placing the original camera clip beside a moving waveform,
spectrogram, and label plot. Prediction media uses the target video and
predicted audio.

Only prediction directories that contain the requested audio filename are
processed, so holdout/missing predictions are skipped naturally.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Iterable

import numpy as np
import soundfile as sf
import torch


DEFAULT_INFER_ROOT = Path("/home/prj/ego2ego_only_mag/inference_result/geocue_True")
DEFAULT_VIDEO_ROOT = Path(
    "/home/prj/data/egocom/480p/5min_parts_10s_stride5_exact10"
)
DEFAULT_AUDIO_ROOT = None
DEFAULT_MANIFEST = Path(
    "/home/prj/ego2ego_only_mag/manifests/egocom_test_pairs_geocue_target_exact10.jsonl"
)
DEFAULT_SPEAKER_LABELS_JSON = Path(
    "/home/prj/data/EgoCom-Dataset/egocom_dataset/speaker_labels/rev_ground_truth_speaker_labels.json"
)
DEFAULT_OUTPUT_ROOT = None
DEFAULT_VIDEO_SCALE = "1280:-2"
DEFAULT_OUTPUT_EXT = ".mp4"
LABEL_BIN_MS = 500

SAMPLE_DIR_RE = re.compile(r"^sample_(\d+)_")
VIDEO_SUFFIXES = (".MP4", ".mp4", ".MOV", ".mov", ".mkv", ".avi")
AUDIO_NAMES = {
    "source": ("src_audio.wav", "*_src_audio.wav"),
    "target": ("tgt_audio.wav", "*_tgt_audio.wav"),
    "pred": ("pred_audio.wav", "*_pred_audio.wav"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create EgoCom source/target/prediction videos with animated "
            "waveform, spectrogram, and label plots."
        )
    )
    parser.add_argument(
        "--infer-root",
        type=Path,
        default=DEFAULT_INFER_ROOT,
        help="Inference output root containing audio/ and plots/ directories.",
    )
    parser.add_argument("--audio-root", type=Path, default=DEFAULT_AUDIO_ROOT)
    parser.add_argument(
        "--plot-root",
        type=Path,
        default=None,
        help="Plot root. Defaults to <infer-root>/plots.",
    )
    parser.add_argument("--video-root", type=Path, default=DEFAULT_VIDEO_ROOT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=(
            "Output root. Defaults to <infer-root>/pred_media, for example "
            "/home/prj/ego2ego_only_mag/inference_result/geocue_True/pred_media."
        ),
    )
    parser.add_argument(
        "--audio-name",
        default="pred_audio.wav",
        help="Predicted audio filename inside each sample directory.",
    )
    parser.add_argument(
        "--media-kind",
        choices=("pred", "source", "target", "all"),
        default="all",
        help=(
            "Explicit media selection. Default is all."
        ),
    )
    parser.add_argument(
        "--output-layout",
        choices=("sample", "video"),
        default="sample",
        help=(
            "Use 'sample' to write one folder per prediction sample with "
            "pred/source/target files. Use 'video' for the legacy layout "
            "grouped by video filename."
        ),
    )
    parser.add_argument(
        "--make-gt-origianl",
        action="store_true",
        help=(
            "Also copy source and target ground-truth MP4s. The original "
            "audio tracks are preserved."
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
        "--audio-codec",
        default="aac",
        help="ffmpeg audio codec for pred media. Use 'copy' only if valid.",
    )
    parser.add_argument(
        "--audio-flipped",
        action="store_true",
        help="Swap predicted audio left/right channels before muxing.",
    )
    parser.add_argument(
        "--audio-bitrate",
        default="192k",
        help="ffmpeg audio bitrate when audio codec is not 'copy'.",
    )
    parser.add_argument(
        "--video-scale",
        default=DEFAULT_VIDEO_SCALE,
        help=(
            "ffmpeg scale size for the camera and plot panels, for example "
            "1280:-2 or 1280x960. Composite output is always re-encoded."
        ),
    )
    parser.add_argument(
        "--video-codec",
        default="libx264",
        help="ffmpeg video codec when --video-scale is enabled.",
    )
    parser.add_argument(
        "--video-crf",
        default="18",
        help="x264/x265 CRF when --video-scale is enabled.",
    )
    parser.add_argument(
        "--video-preset",
        default="medium",
        help="ffmpeg video preset when --video-scale is enabled.",
    )
    parser.add_argument(
        "--speaker-labels-json",
        type=Path,
        default=(
            DEFAULT_SPEAKER_LABELS_JSON
            if DEFAULT_SPEAKER_LABELS_JSON.is_file()
            else None
        ),
        help=(
            "Optional EgoCom speaker-label JSON. When omitted, the label "
            "subplot is rendered as silence. Defaults to the canonical "
            "EgoCom label JSON when it exists."
        ),
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=8,
        help=(
            "Default FPS for both animated plot timing bars and final media. "
            "Override either side with --plot-fps or --output-fps."
        ),
    )
    parser.add_argument(
        "--plot-fps",
        type=int,
        default=None,
        help="FPS for animated plot timing bars. Defaults to --fps.",
    )
    parser.add_argument(
        "--plot-dpi",
        type=int,
        default=80,
        help="DPI for animated plot media.",
    )
    parser.add_argument(
        "--plot-codec",
        default="libx264",
        help="Matplotlib ffmpeg codec for temporary animated plot media.",
    )
    parser.add_argument(
        "--output-fps",
        type=int,
        default=None,
        help="FPS for the final camera/plot media. Defaults to --fps.",
    )
    parser.add_argument(
        "--ext",
        default=DEFAULT_OUTPUT_EXT,
        help="Output extension, for example .mp4.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned mappings without running ffmpeg.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most this many matched predictions.",
    )
    parser.add_argument(
        "--ffmpeg",
        default="ffmpeg",
        help="ffmpeg executable path.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero if a prediction cannot be mapped to a source video.",
    )
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


def iter_prediction_dirs(audio_root: Path, audio_name: str) -> Iterable[tuple[int, Path, Path]]:
    for sample_dir in sorted(audio_root.iterdir()):
        if not sample_dir.is_dir():
            continue
        match = SAMPLE_DIR_RE.match(sample_dir.name)
        if not match:
            continue
        audio_path = sample_dir / audio_name
        if not audio_path.is_file():
            continue
        yield int(match.group(1)), sample_dir, audio_path


def find_video_path(video_root: Path, video_name: str, clip_filename: str) -> Path | None:
    clip_stem = Path(clip_filename).stem
    video_dir = video_root / video_name
    for suffix in VIDEO_SUFFIXES:
        candidate = video_dir / f"{clip_stem}{suffix}"
        if candidate.is_file():
            return candidate
    matches = sorted(video_dir.glob(f"{clip_stem}.*")) if video_dir.is_dir() else []
    return matches[0] if matches else None


def normalize_video_scale(video_scale: str | None) -> str | None:
    if video_scale is None:
        return DEFAULT_VIDEO_SCALE
    video_scale = video_scale.strip()
    if not video_scale or video_scale.lower() in {"none", "copy", "original"}:
        return DEFAULT_VIDEO_SCALE
    if "x" in video_scale and ":" not in video_scale:
        video_scale = video_scale.replace("x", ":", 1)
    return video_scale


def append_video_encoding_args(
    cmd: list[str],
    *,
    video_scale: str | None,
    video_codec: str,
    video_crf: str,
    video_preset: str,
) -> None:
    if video_scale is None:
        cmd.extend(["-c:v", "copy"])
        return

    cmd.extend(
        [
            "-vf",
            f"scale={video_scale}:flags=lanczos,format=yuv420p",
            "-c:v",
            video_codec,
            "-preset",
            video_preset,
            "-crf",
            video_crf,
            "-pix_fmt",
            "yuv420p",
        ]
    )


def build_ffmpeg_cmd(
    ffmpeg: str,
    video_path: Path,
    audio_path: Path,
    output_path: Path,
    audio_codec: str,
    audio_bitrate: str,
    audio_flipped: bool,
    video_scale: str | None,
    video_codec: str,
    video_crf: str,
    video_preset: str,
    overwrite: bool,
) -> list[str]:
    if audio_flipped and audio_codec == "copy":
        raise ValueError("--audio-flipped requires re-encoding audio; do not use --audio-codec copy")

    cmd = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y" if overwrite else "-n",
        "-i",
        str(video_path),
        "-i",
        str(audio_path),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
    ]
    append_video_encoding_args(
        cmd,
        video_scale=video_scale,
        video_codec=video_codec,
        video_crf=video_crf,
        video_preset=video_preset,
    )
    cmd.extend(["-c:a", audio_codec])
    if audio_flipped:
        cmd.extend(["-af", "pan=stereo|c0=c1|c1=c0"])
    cmd.append("-shortest")
    if audio_codec != "copy":
        cmd.extend(["-b:a", audio_bitrate])
    cmd.extend(["-movflags", "+faststart"])
    cmd.append(str(output_path))
    return cmd


def build_video_only_cmd(
    ffmpeg: str,
    video_path: Path,
    output_path: Path,
    video_scale: str | None,
    video_codec: str,
    video_crf: str,
    video_preset: str,
    overwrite: bool,
) -> list[str]:
    cmd = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y" if overwrite else "-n",
        "-i",
        str(video_path),
        "-map",
        "0:v:0",
        "-map",
        "0:a?",
    ]
    append_video_encoding_args(
        cmd,
        video_scale=video_scale,
        video_codec=video_codec,
        video_crf=video_crf,
        video_preset=video_preset,
    )
    cmd.extend(["-c:a", "copy", "-movflags", "+faststart", str(output_path)])
    return cmd


def media_output_path(
    output_root: Path,
    media_kind: str,
    video_path: Path,
    output_ext: str | None,
    sample_dir: Path | None = None,
    output_layout: str = "sample",
) -> Path:
    out_suffix = output_ext or video_path.suffix
    if output_layout == "sample":
        if sample_dir is None:
            raise ValueError("sample_dir is required for sample output layout")
        return output_root / sample_dir.name / f"{media_kind}{out_suffix}"
    return output_root / media_kind / video_path.parent.name / f"{video_path.stem}{out_suffix}"


def mux_media(
    *,
    args: argparse.Namespace,
    media_kind: str,
    video_path: Path,
    audio_path: Path,
    output_path: Path,
    audio_flipped: bool,
) -> tuple[bool, bool]:
    if output_path.exists() and not args.overwrite:
        print(f"SKIP exists: {output_path}", file=sys.stderr)
        return False, False

    print(f"{media_kind}: {audio_path} -> {video_path} -> {output_path}")
    if args.dry_run:
        return True, False

    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        cmd = build_ffmpeg_cmd(
            args.ffmpeg,
            video_path,
            audio_path,
            output_path,
            args.audio_codec,
            args.audio_bitrate,
            audio_flipped,
            args.video_scale,
            args.video_codec,
            args.video_crf,
            args.video_preset,
            args.overwrite,
        )
    except ValueError as exc:
        print(f"ERROR {exc}", file=sys.stderr)
        return False, True

    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"ERROR ffmpeg failed for {media_kind}: {output_path}", file=sys.stderr)
        return False, True
    return True, False


def copy_media(
    *,
    args: argparse.Namespace,
    media_kind: str,
    video_path: Path,
    output_path: Path,
) -> tuple[bool, bool]:
    if output_path.exists() and not args.overwrite:
        print(f"SKIP exists: {output_path}", file=sys.stderr)
        return False, False

    print(f"{media_kind}: {video_path} -> {output_path}")
    if args.dry_run:
        return True, False

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if args.video_scale is not None:
        cmd = build_video_only_cmd(
            args.ffmpeg,
            video_path,
            output_path,
            args.video_scale,
            args.video_codec,
            args.video_crf,
            args.video_preset,
            args.overwrite,
        )
        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            print(f"ERROR ffmpeg failed for {media_kind}: {output_path}", file=sys.stderr)
            return False, True
        return True, False

    shutil.copy2(video_path, output_path)
    return True, False


def resolve_roots(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    audio_root = args.audio_root or args.infer_root / "audio"
    plot_root = args.plot_root or args.infer_root / "plots"
    output_root = args.output_root or args.infer_root / "pred_media"
    return audio_root, plot_root, output_root


def load_speaker_labels(path: Path | None) -> dict | None:
    if path is None:
        return None
    with path.open("r", encoding="utf-8") as handle:
        labels = json.load(handle)
    if not isinstance(labels, dict):
        raise ValueError(f"Expected speaker-label JSON object: {path}")
    return labels


def find_audio_path(sample_dir: Path, media_kind: str, pred_audio_name: str) -> Path | None:
    exact_name, glob_pattern = AUDIO_NAMES[media_kind]
    if media_kind == "pred":
        exact_name = pred_audio_name
    exact_path = sample_dir / exact_name
    if exact_path.is_file():
        return exact_path
    matches = sorted(sample_dir.glob(glob_pattern))
    return matches[0] if matches else None


def load_stereo_audio(path: Path) -> tuple[torch.Tensor, int]:
    audio, sample_rate = sf.read(path, always_2d=True)
    audio = audio.T.astype(np.float32)
    if audio.shape[0] == 1:
        audio = np.repeat(audio, 2, axis=0)
    elif audio.shape[0] > 2:
        audio = audio[:2]
    np.clip(audio, -1.0, 1.0, out=audio)
    return torch.from_numpy(audio), int(sample_rate)


def audio_duration(audio: torch.Tensor, sample_rate: int) -> float:
    return int(audio.shape[-1]) / float(sample_rate)


def map_dominant_speaker_ids_to_roles(
    dominant_speaker_ids: list[int],
    src_person_id: int,
    tgt_person_id: int,
) -> list[int]:
    labels = np.asarray(dominant_speaker_ids, dtype=np.int64)
    mapped = np.full(labels.shape, 2, dtype=np.int64)
    mapped[labels < 0] = -1
    mapped[labels == int(src_person_id)] = 0
    mapped[labels == int(tgt_person_id)] = 1
    return mapped.tolist()


def dominant_speaker_ids_for_plot(
    row: dict,
    speaker_labels: dict | None,
    fallback_duration: float,
) -> torch.Tensor:
    scene_name = row.get("scene_name", "")
    clip_start_ms = int(row.get("clip_start_ms", 0))
    clip_end_ms = int(row.get("clip_end_ms", 0))
    clip_duration_ms = max(0, clip_end_ms - clip_start_ms)
    if clip_duration_ms <= 0:
        clip_duration_ms = max(1, int(round(float(fallback_duration) * 1000.0)))

    label_count = max(1, int(round(clip_duration_ms / LABEL_BIN_MS)))
    if not speaker_labels or scene_name not in speaker_labels:
        return torch.full((label_count,), -1, dtype=torch.long)

    label_start = clip_start_ms // LABEL_BIN_MS
    labels = speaker_labels[scene_name][label_start:label_start + label_count]
    if len(labels) < label_count:
        labels = labels + [-1] * (label_count - len(labels))
    mapped = map_dominant_speaker_ids_to_roles(
        labels,
        int(row.get("src_person_id", -999)),
        int(row.get("tgt_person_id", -998)),
    )
    return torch.tensor(mapped, dtype=torch.long)


def downsample_waveform_for_plot(
    audio: torch.Tensor,
    sample_rate: int,
    max_points: int = 5000,
) -> tuple[np.ndarray, np.ndarray]:
    length = int(audio.shape[-1])
    if length <= max_points:
        indices = torch.arange(length)
    else:
        indices = torch.linspace(0, length - 1, max_points).round().long()
    times = indices.float() / float(sample_rate)
    return times.numpy(), audio[:, indices].numpy()


def plot_waveform_axis(axis, audio: torch.Tensor, sample_rate: int, duration: float) -> None:
    times, audio_plot = downsample_waveform_for_plot(audio, sample_rate)
    limit = max(float(torch.abs(audio).max().item()), 1e-3) * 1.05
    axis.plot(times, audio_plot[0], linewidth=0.6, alpha=0.85, label="L")
    axis.plot(times, audio_plot[1], linewidth=0.6, alpha=0.6, label="R")
    axis.set_title("Waveform")
    axis.set_ylabel("Amp")
    axis.set_xlim(0.0, duration)
    axis.set_ylim(-limit, limit)
    axis.grid(alpha=0.2)
    axis.legend(loc="upper right", fontsize=8)


def plot_spectrogram_axis(
    axis,
    audio: torch.Tensor,
    sample_rate: int,
    title: str,
    duration: float,
    *,
    vmin: float = -120.0,
    vmax: float = 25.0,
):
    window = torch.hann_window(1024)
    spec = torch.stft(
        audio.float(),
        n_fft=1024,
        hop_length=256,
        win_length=1024,
        window=window,
        center=True,
        return_complex=True,
    )
    db = 20.0 * torch.log10(spec.abs().clamp_min(1e-8))
    image = axis.imshow(
        db.numpy(),
        origin="lower",
        aspect="auto",
        extent=[0.0, duration, 0, db.shape[0]],
        cmap="magma",
        vmin=vmin,
        vmax=vmax,
    )
    axis.set_title(title)
    axis.set_ylabel("Freq Bin")
    axis.set_xlim(0.0, duration)
    return image


def plot_label_axis(axis, dominant_speaker_ids: torch.Tensor, duration: float) -> None:
    label_count = int(dominant_speaker_ids.numel())
    label_time = torch.arange(label_count + 1, dtype=torch.float32) * (
        LABEL_BIN_MS / 1000.0
    )
    label_time = torch.clamp(label_time, max=float(duration))
    values = torch.cat([dominant_speaker_ids, dominant_speaker_ids[-1:]])
    if float(label_time[-1].item()) < float(duration):
        label_time = torch.cat([label_time, torch.tensor([float(duration)])])
        values = torch.cat([values, values[-1:]])
    axis.step(label_time.numpy(), values.numpy(), where="post", linewidth=1.4, color="black")
    axis.set_title("Label Cue")
    axis.set_ylabel("Role")
    axis.set_yticks([-1, 0, 1, 2])
    axis.set_yticklabels(["silence", "src", "tgt", "other"])
    axis.set_xlim(0.0, duration)
    axis.set_ylim(-1.5, 2.5)
    axis.grid(alpha=0.2)


def save_stream_plot_media(
    *,
    media_path: Path,
    media_kind: str,
    audio: torch.Tensor,
    sample_rate: int,
    row: dict,
    dominant_speaker_ids: torch.Tensor,
    fps: int,
    dpi: int,
    codec: str,
    ffmpeg: str,
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    matplotlib.rcParams["animation.ffmpeg_path"] = str(ffmpeg)
    import matplotlib.animation as animation
    import matplotlib.pyplot as plt

    if not animation.writers.is_available("ffmpeg"):
        raise RuntimeError("Animated plot media requires the Matplotlib ffmpeg writer")

    duration = audio_duration(audio, sample_rate)
    media_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(
        4,
        1,
        figsize=(12.8, 8.4),
        gridspec_kw={"height_ratios": [0.65, 1.0, 1.15, 1.15]},
        constrained_layout=True,
    )
    clip_key = {
        "source": "src_clip_filename",
        "target": "tgt_clip_filename",
        "pred": "tgt_clip_filename",
    }[media_kind]
    fig.suptitle(
        f"{media_kind.capitalize()} | {row.get('scene_name', '')} | "
        f"src={row.get('src_person_id', '')} tgt={row.get('tgt_person_id', '')} | "
        f"{row.get(clip_key, '')}",
        fontsize=11,
    )

    plot_label_axis(axes[0], dominant_speaker_ids, duration)
    plot_waveform_axis(axes[1], audio, sample_rate, duration)
    image = plot_spectrogram_axis(axes[2], audio[0], sample_rate, "Spectrogram (L)", duration)
    plot_spectrogram_axis(axes[3], audio[1], sample_rate, "Spectrogram (R)", duration)
    axes[-1].set_xlabel("Time (s)")
    colorbar = fig.colorbar(image, ax=axes[2:], shrink=0.94)
    colorbar.set_label("dB")

    bars = []
    for axis in axes:
        if axis in (axes[2], axes[3]):
            bars.append(axis.axvline(0.0, color="black", linewidth=3.0, alpha=0.75))
            bars.append(axis.axvline(0.0, color="white", linewidth=1.5, alpha=1.0))
        else:
            bars.append(axis.axvline(0.0, color="red", linewidth=1.8, alpha=0.95))

    frame_count = max(2, int(round(duration * float(fps))) + 1)
    frame_times = torch.linspace(0.0, float(duration), frame_count).tolist()

    def update(frame_time):
        for bar in bars:
            bar.set_xdata([frame_time, frame_time])
        return bars

    try:
        anim = animation.FuncAnimation(
            fig,
            update,
            frames=frame_times,
            interval=1000.0 / float(fps),
            blit=False,
            repeat=False,
        )
        extra_args = ["-pix_fmt", "yuv420p", "-movflags", "+faststart"]
        if str(codec) == "libx264":
            extra_args[2:2] = ["-profile:v", "baseline", "-level", "3.0"]
        writer = animation.FFMpegWriter(fps=fps, codec=codec, extra_args=extra_args)
        anim.save(media_path, writer=writer, dpi=dpi)
    finally:
        plt.close(fig)

    return media_path


def build_composite_cmd(
    *,
    ffmpeg: str,
    camera_video_path: Path,
    plot_video_path: Path,
    output_path: Path,
    video_scale: str,
    video_codec: str,
    video_crf: str,
    video_preset: str,
    overwrite: bool,
    audio_path: Path | None,
    audio_flipped: bool,
    audio_codec: str,
    audio_bitrate: str,
    duration: float,
    output_fps: int,
) -> list[str]:
    if audio_flipped and audio_codec == "copy":
        raise ValueError("--audio-flipped requires re-encoding audio; do not use --audio-codec copy")

    cmd = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y" if overwrite else "-n",
        "-i",
        str(camera_video_path),
        "-i",
        str(plot_video_path),
    ]
    if audio_path is not None:
        cmd.extend(["-i", str(audio_path)])

    filter_complex = (
        f"[0:v]scale={video_scale}:flags=lanczos,setsar=1[v0];"
        "[1:v][v0]scale2ref=w=-2:h=ih[p0][v0ref];"
        "[v0ref][p0]hstack=inputs=2:shortest=1[stacked];"
        f"[stacked]fps={int(output_fps)}[vout]"
    )
    cmd.extend(["-filter_complex", filter_complex, "-map", "[vout]"])

    if audio_path is None:
        cmd.extend(["-map", "0:a?", "-c:a", "copy"])
    else:
        cmd.extend(["-map", "2:a:0", "-c:a", audio_codec])
        if audio_flipped:
            cmd.extend(["-af", "pan=stereo|c0=c1|c1=c0"])
        if audio_codec != "copy":
            cmd.extend(["-b:a", audio_bitrate])

    cmd.extend(
        [
            "-c:v",
            video_codec,
            "-preset",
            video_preset,
            "-crf",
            video_crf,
            "-pix_fmt",
            "yuv420p",
            "-t",
            f"{float(duration):.6f}",
            "-movflags",
            "+faststart",
            str(output_path),
        ]
    )
    return cmd


def make_composite_media(
    *,
    args: argparse.Namespace,
    media_kind: str,
    row: dict,
    camera_video_path: Path,
    plot_audio_path: Path,
    output_path: Path,
    speaker_labels: dict | None,
    pred_audio_path: Path | None,
) -> tuple[bool, bool]:
    if output_path.exists() and not args.overwrite:
        print(f"SKIP exists: {output_path}", file=sys.stderr)
        return False, False

    print(f"{media_kind}: {camera_video_path} + {plot_audio_path} -> {output_path}")
    if args.dry_run:
        return True, False

    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        audio, sample_rate = load_stereo_audio(plot_audio_path)
        duration = audio_duration(audio, sample_rate)
        dominant_speaker_ids = dominant_speaker_ids_for_plot(row, speaker_labels, duration)
        with tempfile.TemporaryDirectory(prefix="make_pred_media_") as temp_dir:
            temp_plot_path = Path(temp_dir) / f"{media_kind}_plot.mp4"
            save_stream_plot_media(
                media_path=temp_plot_path,
                media_kind=media_kind,
                audio=audio,
                sample_rate=sample_rate,
                row=row,
                dominant_speaker_ids=dominant_speaker_ids,
                fps=args.plot_fps,
                dpi=args.plot_dpi,
                codec=args.plot_codec,
                ffmpeg=args.ffmpeg,
            )
            cmd = build_composite_cmd(
                ffmpeg=args.ffmpeg,
                camera_video_path=camera_video_path,
                plot_video_path=temp_plot_path,
                output_path=output_path,
                video_scale=args.video_scale,
                video_codec=args.video_codec,
                video_crf=args.video_crf,
                video_preset=args.video_preset,
                overwrite=args.overwrite,
                audio_path=pred_audio_path,
                audio_flipped=args.audio_flipped if media_kind == "pred" else False,
                audio_codec=args.audio_codec,
                audio_bitrate=args.audio_bitrate,
                duration=duration,
                output_fps=args.output_fps,
            )
            result = subprocess.run(cmd, check=False)
            if result.returncode != 0:
                print(f"ERROR ffmpeg failed for {media_kind}: {output_path}", file=sys.stderr)
                return False, True
    except Exception as exc:  # pylint: disable=broad-except
        print(f"ERROR {media_kind} {output_path}: {exc}", file=sys.stderr)
        return False, True

    return True, False


def main() -> int:
    args = parse_args()
    if args.fps <= 0:
        raise ValueError(f"--fps must be positive, got {args.fps}")
    if args.plot_fps is None:
        args.plot_fps = args.fps
    if args.output_fps is None:
        args.output_fps = args.fps
    if args.plot_fps <= 0 or args.output_fps <= 0:
        raise ValueError(
            f"--plot-fps and --output-fps must be positive, got "
            f"{args.plot_fps} and {args.output_fps}"
        )
    args.video_scale = normalize_video_scale(args.video_scale)
    if args.ext is None:
        args.ext = DEFAULT_OUTPUT_EXT
    if not args.ext.startswith("."):
        args.ext = f".{args.ext}"

    audio_root, plot_root, output_root = resolve_roots(args)
    if not audio_root.is_dir():
        raise FileNotFoundError(f"Audio root does not exist: {audio_root}")
    if not plot_root.is_dir():
        print(
            f"WARN plot root does not exist; generating animated plots from audio: {plot_root}",
            file=sys.stderr,
        )

    rows = load_manifest(args.manifest)
    speaker_labels = load_speaker_labels(args.speaker_labels_json)

    make_pred = args.media_kind in ("all", "pred")
    make_gt_original = args.make_gt_origianl or args.media_kind == "all"
    make_source = make_gt_original or args.media_kind == "source"
    make_target = make_gt_original or args.media_kind == "target"

    processed = 0
    skipped = 0
    failed = 0

    for sample_index, sample_dir, audio_path in iter_prediction_dirs(
        audio_root, args.audio_name
    ):
        manifest_index = sample_index - args.sample_index_base
        if manifest_index < 0 or manifest_index >= len(rows):
            print(
                f"SKIP no manifest row for {sample_dir.name} "
                f"(computed index {manifest_index})",
                file=sys.stderr,
            )
            skipped += 1
            if args.strict:
                failed += 1
            continue

        row = rows[manifest_index]
        outputs_done = 0
        sample_failed = False

        src_audio_path = find_audio_path(sample_dir, "source", args.audio_name)
        tgt_audio_path = find_audio_path(sample_dir, "target", args.audio_name)
        if make_source and src_audio_path is None:
            print(f"SKIP missing source audio for {sample_dir.name}", file=sys.stderr)
            skipped += 1
            sample_failed = args.strict
        if (make_target or make_pred) and tgt_audio_path is None:
            print(f"SKIP missing target audio for {sample_dir.name}", file=sys.stderr)
            skipped += 1
            sample_failed = args.strict

        tgt_video_path = None
        if make_pred or make_target:
            tgt_video_path = find_video_path(
                args.video_root,
                row["tgt_video_name"],
                row["tgt_clip_filename"],
            )
            if tgt_video_path is None:
                print(
                    f"SKIP missing target video for {sample_dir.name}: "
                    f"{args.video_root / row['tgt_video_name'] / Path(row['tgt_clip_filename']).with_suffix('.MP4').name}",
                    file=sys.stderr,
                )
                skipped += 1
                sample_failed = args.strict

        src_video_path = None
        if make_source:
            src_video_path = find_video_path(
                args.video_root,
                row["src_video_name"],
                row["src_clip_filename"],
            )
            if src_video_path is None:
                print(
                    f"SKIP missing source video for {sample_dir.name}: "
                    f"{args.video_root / row['src_video_name'] / Path(row['src_clip_filename']).with_suffix('.MP4').name}",
                    file=sys.stderr,
                )
                skipped += 1
                sample_failed = args.strict

        if make_pred and tgt_video_path is not None:
            ok, did_fail = make_composite_media(
                args=args,
                media_kind="pred",
                row=row,
                camera_video_path=tgt_video_path,
                plot_audio_path=audio_path,
                output_path=media_output_path(
                    output_root,
                    "pred",
                    tgt_video_path,
                    args.ext,
                    sample_dir=sample_dir,
                    output_layout=args.output_layout,
                ),
                speaker_labels=speaker_labels,
                pred_audio_path=audio_path,
            )
            outputs_done += int(ok)
            failed += int(did_fail)
            sample_failed = sample_failed or did_fail

        if make_target and tgt_video_path is not None and tgt_audio_path is not None:
            ok, did_fail = make_composite_media(
                args=args,
                media_kind="target",
                row=row,
                camera_video_path=tgt_video_path,
                plot_audio_path=tgt_audio_path,
                output_path=media_output_path(
                    output_root,
                    "target",
                    tgt_video_path,
                    args.ext,
                    sample_dir=sample_dir,
                    output_layout=args.output_layout,
                ),
                speaker_labels=speaker_labels,
                pred_audio_path=None,
            )
            outputs_done += int(ok)
            failed += int(did_fail)
            sample_failed = sample_failed or did_fail

        if make_source and src_video_path is not None and src_audio_path is not None:
            ok, did_fail = make_composite_media(
                args=args,
                media_kind="source",
                row=row,
                camera_video_path=src_video_path,
                plot_audio_path=src_audio_path,
                output_path=media_output_path(
                    output_root,
                    "source",
                    src_video_path,
                    args.ext,
                    sample_dir=sample_dir,
                    output_layout=args.output_layout,
                ),
                speaker_labels=speaker_labels,
                pred_audio_path=None,
            )
            outputs_done += int(ok)
            failed += int(did_fail)
            sample_failed = sample_failed or did_fail

        if sample_failed and args.strict:
            break
        if outputs_done == 0:
            skipped += 1
            continue

        processed += 1
        if args.limit is not None and processed >= args.limit:
            break

    print(f"done: processed={processed} skipped={skipped} failed={failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
