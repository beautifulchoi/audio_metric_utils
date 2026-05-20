from pathlib import Path

import torch


def _plot_duration_seconds(src_audio, tgt_audio, pred_audio, sample_rate):
    max_samples = max(
        int(src_audio.shape[-1]),
        int(tgt_audio.shape[-1]),
        int(pred_audio.shape[-1]),
    )
    return max_samples / float(sample_rate)


def _plot_spectrogram(axis, audio, sample_rate, title, vmin=-120.0, vmax=25.0):
    spec = torch.stft(
        audio.float(),
        n_fft=1024,
        hop_length=256,
        win_length=1024,
        window=torch.hann_window(1024),
        center=True,
        return_complex=True,
    )
    magnitude = spec.abs().clamp_min(1e-8)
    db = 20.0 * torch.log10(magnitude)
    image = axis.imshow(
        db.numpy(),
        origin="lower",
        aspect="auto",
        extent=[0, audio.numel() / float(sample_rate), 0, db.shape[0]],
        cmap="magma",
        vmin=vmin,
        vmax=vmax,
    )
    axis.set_title(title)
    axis.set_ylabel("Freq Bin")
    return image


def _reference_waveform_ylim(src_audio, tgt_audio):
    reference = torch.cat([src_audio.reshape(-1), tgt_audio.reshape(-1)])
    y_min = float(reference.min().item())
    y_max = float(reference.max().item())
    if y_min == y_max:
        margin = max(abs(y_min), 1e-3) * 0.05
        return y_min - margin, y_max + margin
    margin = (y_max - y_min) * 0.05
    return y_min - margin, y_max + margin


def _plot_waveform(axis, audio, sample_rate, title, ylim):
    length = audio.shape[-1]
    time = torch.arange(length, dtype=torch.float32) / float(sample_rate)
    axis.plot(time.numpy(), audio[0].numpy(), linewidth=0.5, alpha=0.8, label="L")
    axis.plot(time.numpy(), audio[1].numpy(), linewidth=0.5, alpha=0.5, label="R")
    axis.set_title(title)
    axis.set_ylabel("Amp")
    axis.set_ylim(*ylim)
    axis.grid(alpha=0.2)
    axis.legend(loc="upper right", fontsize=8)
    axis.set_xlim(0, length / float(sample_rate))


def _plot_label(axis, dominant_speaker_ids, duration):
    labels = dominant_speaker_ids.long().reshape(-1)
    if labels.numel() == 0:
        axis.text(0.5, 0.5, "No labels", ha="center", va="center", transform=axis.transAxes)
        axis.set_xlim(0.0, float(duration))
    else:
        label_time = torch.linspace(0.0, float(duration), labels.numel() + 1)
        label_values = torch.cat([labels, labels[-1:]])
        axis.step(
            label_time.numpy(),
            label_values.numpy(),
            where="post",
            linewidth=1.4,
            color="black",
        )
        axis.set_xlim(0.0, float(duration))
    axis.set_title("Label Cue")
    axis.set_ylabel("Role")
    axis.set_yticks([-1, 0, 1, 2])
    axis.set_yticklabels(["silence", "src", "tgt", "other"])
    axis.set_ylim(-1.5, 2.5)
    axis.grid(alpha=0.2)


def _add_timing_bars(axes, duration):
    bars = []
    for axis in [axes[0, 0], *axes[1:, 0].ravel().tolist()]:
        axis.set_xlim(0.0, duration)
        bars.append(axis.axvline(0.0, color="red", linewidth=1.6, alpha=0.95))
    for axis in axes[1:, 1:].ravel().tolist():
        axis.set_xlim(0.0, duration)
        bars.append(axis.axvline(0.0, color="black", linewidth=3.0, alpha=0.9))
        bars.append(axis.axvline(0.0, color="white", linewidth=1.5, alpha=1.0))
    return bars


def _save_timing_bar_media(
    fig,
    axes,
    media_path,
    duration,
    fps,
    dpi,
    codec,
):
    import matplotlib.animation as animation

    if not animation.writers.is_available("ffmpeg"):
        raise RuntimeError("make_media=True requires the Matplotlib ffmpeg animation writer")

    media_path = Path(media_path)
    media_path.parent.mkdir(parents=True, exist_ok=True)
    bars = _add_timing_bars(axes, duration)
    frame_count = max(2, int(round(duration * float(fps))))
    frame_times = torch.linspace(0.0, float(duration), frame_count).tolist()

    def update(frame_time):
        for bar in bars:
            bar.set_xdata([frame_time, frame_time])
        return bars

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
    writer = animation.FFMpegWriter(
        fps=fps,
        codec=codec,
        extra_args=extra_args,
    )
    anim.save(media_path, writer=writer, dpi=dpi)
    return media_path


def save_sample_plot(
    plot_path,
    src_audio,
    tgt_audio,
    pred_audio,
    dominant_speaker_ids,
    sample_rate,
    title=None,
    *,
    make_media=False,
    media_path=None,
    media_fps=10,
    media_dpi=60,
    media_codec="libx264",
    save_image=True,
):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_path = Path(plot_path)
    plot_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(
        4,
        3,
        figsize=(20, 14),
        gridspec_kw={"width_ratios": [1.6, 1.0, 1.0], "height_ratios": [0.7, 1.0, 1.0, 1.0]},
        constrained_layout=True,
    )
    if title:
        fig.suptitle(title, fontsize=12)

    duration = _plot_duration_seconds(src_audio, tgt_audio, pred_audio, sample_rate)
    _plot_label(axes[0, 0], dominant_speaker_ids, duration)
    axes[0, 1].axis("off")
    axes[0, 2].axis("off")

    rows = [
        ("Source", src_audio),
        ("Target", tgt_audio),
        ("Predicted", pred_audio),
    ]
    waveform_ylim = _reference_waveform_ylim(src_audio, tgt_audio)
    image = None
    for row_idx, (name, audio) in enumerate(rows):
        axis_row = row_idx + 1
        _plot_waveform(
            axes[axis_row, 0],
            audio,
            sample_rate,
            f"{name} Waveform",
            waveform_ylim,
        )
        image = _plot_spectrogram(
            axes[axis_row, 1],
            audio[0],
            sample_rate,
            f"{name} Spectrogram (L)",
        )
        image = _plot_spectrogram(
            axes[axis_row, 2],
            audio[1],
            sample_rate,
            f"{name} Spectrogram (R)",
        )

    for axis in axes[-1, :]:
        axis.set_xlabel("Time (s)")
    if image is not None:
        colorbar = fig.colorbar(image, ax=axes[1:, 1:].ravel().tolist(), shrink=0.95)
        colorbar.set_label("dB")

    output_media_path = None
    try:
        if save_image:
            fig.savefig(plot_path, dpi=120)
        if make_media:
            if media_path is None:
                media_path = plot_path.with_suffix(".mp4")
            output_media_path = _save_timing_bar_media(
                fig,
                axes,
                media_path,
                duration,
                media_fps,
                media_dpi,
                media_codec,
            )
    finally:
        plt.close(fig)

    return output_media_path
