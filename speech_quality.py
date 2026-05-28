"""Speech quality metrics for saved prediction/target audio pairs."""

from __future__ import annotations

import torch
import torchaudio
from pesq import PesqError
from pesq import pesq as pesq_score
from torchmetrics.audio.sdr import ScaleInvariantSignalDistortionRatio
from torchmetrics.audio.snr import ScaleInvariantSignalNoiseRatio, SignalNoiseRatio


class SpeechQualityMetrics:
    """Compute speech quality metrics on channel-first audio tensors."""

    def __init__(
        self,
        sample_rate: int,
        pesq_max_seconds: float = 30.0,
        pesq_sample_rate: int = 16000,
        pesq_mode: str = "wb",
    ) -> None:
        self.sample_rate = int(sample_rate)
        self.pesq_max_seconds = float(pesq_max_seconds)
        self.pesq_sample_rate = int(pesq_sample_rate)
        self.pesq_mode = pesq_mode
        self.snr = SignalNoiseRatio()
        self.si_snr = ScaleInvariantSignalNoiseRatio()
        self.si_sdr = ScaleInvariantSignalDistortionRatio()
        self._stoi = None

    @property
    def names(self) -> tuple[str, ...]:
        return ("pesq", "stoi", "seg_snr", "si_snr", "si_sdr")

    def compute(self, name: str, pred: torch.Tensor, gt: torch.Tensor) -> float:
        pred, gt = self._match_length(pred.float(), gt.float())

        if name == "pesq":
            return self._pesq(pred, gt)
        if name == "stoi":
            return self._tensor_to_float(self._stoi_metric(pred, gt))
        if name == "seg_snr":
            return self._tensor_to_float(self.snr(pred, gt))
        if name == "si_snr":
            return self._tensor_to_float(self.si_snr(pred, gt))
        if name == "si_sdr":
            return self._tensor_to_float(self.si_sdr(pred, gt))

        raise KeyError(f"Unknown speech quality metric: {name}")

    def _stoi_metric(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        if self._stoi is None:
            try:
                from torchmetrics.audio.stoi import ShortTimeObjectiveIntelligibility

                self._stoi = ShortTimeObjectiveIntelligibility(self.sample_rate)
            except ModuleNotFoundError as exc:
                raise ModuleNotFoundError(
                    "STOI requires pystoi. Install dependencies from "
                    "comparision_baselines_new/DAVIS/requirements.txt or run "
                    "`pip install pystoi==0.4.1`."
                ) from exc
        return self._stoi(pred, gt)

    def _pesq(self, pred: torch.Tensor, gt: torch.Tensor) -> float:
        pred = self._resample_if_needed(pred, self.sample_rate, self.pesq_sample_rate).float()
        gt = self._resample_if_needed(gt, self.sample_rate, self.pesq_sample_rate).float()
        pred, gt = self._match_length(pred, gt)

        weighted_scores = []
        for pred_window, gt_window in self._iter_pesq_windows(pred, gt):
            skip, reason = self._should_skip_pesq(pred_window, gt_window)
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
                        self.pesq_sample_rate,
                        gt_np[channel],
                        pred_np[channel],
                        self.pesq_mode,
                    )
                except PesqError as exc:
                    print(f"[SKIP PESQ CHANNEL] {exc}")
                    continue
                weighted_scores.append((float(score), window_samples))

        if not weighted_scores:
            raise RuntimeError("skip_pesq:no_valid_windows")

        total_samples = sum(samples for _, samples in weighted_scores)
        return float(sum(score * samples for score, samples in weighted_scores) / total_samples)

    def _iter_pesq_windows(self, pred: torch.Tensor, gt: torch.Tensor) -> list[tuple[torch.Tensor, torch.Tensor]]:
        pred, gt = self._match_length(pred, gt)
        if self.pesq_max_seconds <= 0:
            return [(pred, gt)]

        window_size = int(self.pesq_max_seconds * self.pesq_sample_rate)
        min_len = int(0.25 * self.pesq_sample_rate)
        if window_size <= min_len:
            raise ValueError("--pesq-max-seconds must be greater than 0.25 seconds")

        windows = []
        for start in range(0, pred.shape[-1], window_size):
            end = min(start + window_size, pred.shape[-1])
            if end - start < min_len:
                continue
            windows.append((pred[..., start:end], gt[..., start:end]))
        return windows

    def _should_skip_pesq(self, pred: torch.Tensor, gt: torch.Tensor) -> tuple[bool, str]:
        min_len = int(0.25 * self.pesq_sample_rate)

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

    @staticmethod
    def _resample_if_needed(wav: torch.Tensor, orig_sr: int, target_sr: int) -> torch.Tensor:
        if orig_sr == target_sr:
            return wav
        return torchaudio.functional.resample(wav, orig_sr, target_sr)

    @staticmethod
    def _match_length(pred: torch.Tensor, gt: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        min_len = min(pred.shape[-1], gt.shape[-1])
        return pred[..., :min_len], gt[..., :min_len]

    @staticmethod
    def _tensor_to_float(score) -> float:
        if isinstance(score, torch.Tensor):
            score = score.detach().cpu().item()
        return float(score)
