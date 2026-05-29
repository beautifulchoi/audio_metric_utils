# reference : https://github.com/AaronZ345/ISDrama/blob/main
"""Spatialization metrics based on interaural phase and level differences."""

from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from tqdm.auto import tqdm


class DFTBase(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def dft_matrix(n: int) -> np.ndarray:
        x, y = np.meshgrid(np.arange(n), np.arange(n))
        omega = np.exp(-2 * np.pi * 1j / n)
        return np.power(omega, x * y)


def pad_center(data: np.ndarray, *, size: int, axis: int = -1, **kwargs: Any) -> np.ndarray:
    kwargs.setdefault("mode", "constant")
    n = data.shape[axis]
    lpad = int((size - n) // 2)
    if lpad < 0:
        raise ValueError(f"Target size ({size:d}) must be at least input size ({n:d})")

    lengths = [(0, 0)] * data.ndim
    lengths[axis] = (lpad, int(size - n - lpad))
    return np.pad(data, lengths, **kwargs)


class STFT(DFTBase):
    """PyTorch STFT implementation with frozen Conv1d kernels."""

    def __init__(
        self,
        n_fft: int = 2048,
        hop_length: int | None = None,
        win_length: int | None = None,
        window: str = "hann",
        center: bool = True,
        pad_mode: str = "reflect",
        freeze_parameters: bool = True,
    ) -> None:
        super().__init__()
        if pad_mode not in {"constant", "reflect"}:
            raise ValueError("pad_mode must be 'constant' or 'reflect'")

        self.n_fft = n_fft
        self.hop_length = hop_length or int((win_length or n_fft) // 4)
        self.win_length = win_length or n_fft
        self.center = center
        self.pad_mode = pad_mode

        fft_window = librosa.filters.get_window(window, self.win_length, fftbins=True)
        fft_window = pad_center(fft_window, size=n_fft)
        dft_matrix = self.dft_matrix(n_fft)
        out_channels = n_fft // 2 + 1

        self.conv_real = nn.Conv1d(1, out_channels, kernel_size=n_fft, stride=self.hop_length, bias=False)
        self.conv_imag = nn.Conv1d(1, out_channels, kernel_size=n_fft, stride=self.hop_length, bias=False)
        self.conv_real.weight.data = torch.tensor(
            np.real(dft_matrix[:, :out_channels] * fft_window[:, None]).T[:, None, :],
            dtype=torch.float32,
        )
        self.conv_imag.weight.data = torch.tensor(
            np.imag(dft_matrix[:, :out_channels] * fft_window[:, None]).T[:, None, :],
            dtype=torch.float32,
        )

        if freeze_parameters:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, audio: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return real and imaginary tensors shaped (batch, 1, frames, freq_bins)."""
        x = audio[:, None, :]
        if self.center:
            x = F.pad(x, pad=(self.n_fft // 2, self.n_fft // 2), mode=self.pad_mode)

        real = self.conv_real(x)[:, None, :, :].transpose(2, 3)
        imag = self.conv_imag(x)[:, None, :, :].transpose(2, 3)
        return real, imag


class LogmelFilterBank(nn.Module):
    """Frozen mel filter bank matching the provided spatial feature code."""

    def __init__(
        self,
        sr: int = 32000,
        n_fft: int = 1024,
        n_mels: int = 128,
        fmin: float = 50.0,
        fmax: float | None = 14000.0,
        freeze_parameters: bool = True,
    ) -> None:
        super().__init__()
        mel = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax).T
        self.melW = nn.Parameter(torch.tensor(mel, dtype=torch.float32))
        if freeze_parameters:
            for param in self.parameters():
                param.requires_grad = False


class SpatialMetrics:
    """Calculate IPD/ILD spatialization metrics from stereo tensors."""

    def __init__(
        self,
        sample_rate: int = 32000,
        n_fft: int = 1024,
        hop_length: int = 320,
        win_length: int = 1024,
        n_mels: int = 128,
        fmin: float = 50.0,
        fmax: float = 14000.0,
        epsilon: float = 1e-10,
    ) -> None:
        self.sample_rate = int(sample_rate)
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.win_length = int(win_length)
        self.n_mels = int(n_mels)
        self.epsilon = float(epsilon)
        self.stft = STFT(
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window="hann",
            center=True,
            pad_mode="reflect",
            freeze_parameters=True,
        )
        self.logmel = LogmelFilterBank(
            sr=self.sample_rate,
            n_fft=self.n_fft,
            n_mels=self.n_mels,
            fmin=fmin,
            fmax=fmax,
            freeze_parameters=True,
        )

    @property
    def names(self) -> tuple[str, ...]:
        return ("ipd_mae_x10000", "ild_mae_x10000")

    def compute(self, pred: torch.Tensor, gt: torch.Tensor, sample_rate: int) -> dict[str, float]:
        pred_features = self.extract_features(pred, sample_rate)
        gt_features = self.extract_features(gt, sample_rate)
        return {
            "ipd_mae_x10000": self._mae_time_scaled(gt_features["IPD"], pred_features["IPD"]),
            "ild_mae_x10000": self._mae_time_scaled(gt_features["ILD"], pred_features["ILD"]),
        }

    def extract_features(self, waveform: torch.Tensor, sample_rate: int) -> dict[str, np.ndarray]:
        waveform = self._resample_if_needed(waveform.float(), int(sample_rate))
        if waveform.ndim != 2 or waveform.shape[0] != 2:
            raise ValueError(f"Spatial metrics require stereo audio shaped (2, samples), got {tuple(waveform.shape)}")

        waveform = waveform.unsqueeze(0)
        batch, channels, samples = waveform.shape
        waveform_flat = waveform.reshape(batch * channels, samples)

        real, imag = self.stft(waveform_flat)
        real = real[:, 0, :, :]
        imag = imag[:, 0, :, :]

        real_left, real_right = real[0], real[1]
        imag_left, imag_right = imag[0], imag[1]

        phase_left = torch.atan2(imag_left, real_left)
        phase_right = torch.atan2(imag_right, real_right)
        ipd = torch.angle(torch.exp(1j * (phase_right - phase_left)))

        mag_left = torch.sqrt(real_left**2 + imag_left**2)
        mag_right = torch.sqrt(real_right**2 + imag_right**2)
        ild = 20.0 * torch.log10((mag_right + self.epsilon) / (mag_left + self.epsilon))

        melW = self.logmel.melW.to(ipd.device, dtype=ipd.dtype)
        ipd_mel = torch.matmul(ipd, melW)
        ild_mel = torch.matmul(ild, melW)

        return {
            "IPD": ipd_mel.detach().cpu().numpy(),
            "ILD": ild_mel.detach().cpu().numpy(),
        }

    def extract_feature_file(self, audio_path: str | Path, output_dir: str | Path) -> tuple[np.ndarray, np.ndarray]:
        audio_path = Path(audio_path)
        output_dir = Path(output_dir)
        waveform, sample_rate = torchaudio.load(str(audio_path))
        features = self.extract_features(waveform, sample_rate)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / audio_path.name.replace(".wav", "_feature.npy")
        np.save(output_path, features)
        return features["IPD"], features["ILD"]

    def _resample_if_needed(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        if sample_rate == self.sample_rate:
            return waveform
        return torchaudio.functional.resample(waveform, sample_rate, self.sample_rate)

    @staticmethod
    def _mae_time_scaled(matrix_a: np.ndarray, matrix_b: np.ndarray) -> float:
        a = np.asarray(matrix_a)
        b = np.asarray(matrix_b)
        min_rows = min(a.shape[0], b.shape[0])
        min_cols = min(a.shape[1], b.shape[1])
        if min_rows == 0 or min_cols == 0:
            return float("nan")
        mae = np.mean(np.abs(a[:min_rows, :min_cols] - b[:min_rows, :min_cols]))
        return float(mae / min_rows * 10000.0)


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract IPD and ILD features from stereo WAV files.")
    parser.add_argument("--audio-dir", type=Path, required=True, help="Directory containing audio files.")
    parser.add_argument("--output-dir", type=Path, default=Path("./features"), help="Directory to save features.")
    args = parser.parse_args()

    metric = SpatialMetrics()
    audio_files = sorted(glob.glob(os.path.join(str(args.audio_dir), "*.wav")))
    for audio_file in tqdm(audio_files, desc="extracting ipd and ild"):
        metric.extract_feature_file(audio_file, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
