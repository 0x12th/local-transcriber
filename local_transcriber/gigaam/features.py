"""NumPy mel features for the local GigaAM backend."""

from __future__ import annotations

from typing import Any

import numpy as np

from local_transcriber.gigaam.audio import SAMPLE_RATE


def _hz_to_mel(freq: Any) -> Any:
    return 2595.0 * np.log10(1.0 + freq / 700.0)


def _mel_to_hz(mels: Any) -> Any:
    return 700.0 * (10.0 ** (mels / 2595.0) - 1.0)


def _mel_filterbank(
    n_freqs: int, f_min: float, f_max: float, n_mels: int, sample_rate: int
) -> np.ndarray:
    all_freqs = np.linspace(0, sample_rate // 2, n_freqs)
    mel_points = np.linspace(_hz_to_mel(f_min), _hz_to_mel(f_max), n_mels + 2)
    freq_points = _mel_to_hz(mel_points)
    freq_diff = freq_points[1:] - freq_points[:-1]
    slopes = freq_points[np.newaxis, :] - all_freqs[:, np.newaxis]
    down = -slopes[:, :-2] / freq_diff[np.newaxis, :-1]
    up = slopes[:, 2:] / freq_diff[np.newaxis, 1:]
    return np.maximum(0.0, np.minimum(down, up))


class Features:
    def __init__(self, config: dict[str, Any]) -> None:
        preprocessor = config["preprocessor"]
        self.sample_rate = int(preprocessor.get("sample_rate", SAMPLE_RATE))
        self.n_mels = int(preprocessor["features"])
        self.n_fft = int(preprocessor.get("n_fft", self.sample_rate // 40))
        self.win_length = int(
            preprocessor.get("win_length", self.sample_rate // 40)
        )
        self.hop_length = int(
            preprocessor.get("hop_length", self.sample_rate // 100)
        )
        self.center = bool(preprocessor.get("center", True))
        n = np.arange(self.win_length)
        self.window = (
            0.5 - 0.5 * np.cos(2.0 * np.pi * n / self.win_length)
        ).astype(np.float32)
        self.filterbank = _mel_filterbank(
            self.n_fft // 2 + 1,
            0.0,
            self.sample_rate / 2.0,
            self.n_mels,
            self.sample_rate,
        ).astype(np.float32)

    def out_len(self, samples: int) -> int:
        if self.center:
            return max(0, samples // self.hop_length + 1)
        return max(0, (samples - self.win_length) // self.hop_length + 1)

    def __call__(self, wav: np.ndarray) -> np.ndarray:
        audio = np.asarray(wav, dtype=np.float32)
        if self.center:
            if len(audio) == 0:
                return np.empty((1, self.n_mels, 0), dtype=np.float32)
            pad = self.n_fft // 2
            audio = np.pad(audio, (pad, pad), mode="reflect")
        frame_count = max(0, (len(audio) - self.n_fft) // self.hop_length + 1)
        if frame_count == 0:
            return np.empty((1, self.n_mels, 0), dtype=np.float32)
        indices = np.arange(self.n_fft)[np.newaxis, :] + self.hop_length * np.arange(
            frame_count
        )[:, np.newaxis]
        frames = (audio[indices] * self.window[np.newaxis, :]).astype(np.float32)
        spectrum = (
            np.abs(np.fft.rfft(frames, n=self.n_fft, axis=1)) ** 2
        ).astype(np.float32)
        mel = spectrum @ self.filterbank
        mel = np.log(np.clip(mel, 1e-9, 1e9))
        return mel.T[np.newaxis, :, :].astype(np.float32)
