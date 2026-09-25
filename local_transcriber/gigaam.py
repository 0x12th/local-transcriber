"""Local GigaAM v3 ONNX backend adapted from the pinned Giga Pisar core.

Source and license details are recorded in THIRD_PARTY_NOTICES.md.
"""

from __future__ import annotations

import subprocess
import tempfile
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from local_transcriber.model_validation import (
    BUNDLE,
    LoadedModel,
    load_model,
    resolve_model_dir,
)
from local_transcriber.transcript import TranscriptResult

SAMPLE_RATE = 16_000
MODEL_NAME = BUNDLE.profile
MAX_CHUNK = 24.0
MAX_CHUNK_SAMPLES = int(MAX_CHUNK * SAMPLE_RATE)
MIN_PAUSE_OFFSET_SAMPLES = 3 * SAMPLE_RATE
SILENCE_DB = -35
SILENCE_MIN = 0.3
MAX_SYMBOLS_PER_FRAME = 3


@dataclass(frozen=True)
class GigaAMSegment:
    start: float
    end: float
    text: str


def _error_detail(error: subprocess.CalledProcessError) -> str:
    stderr = error.stderr.strip() if isinstance(error.stderr, str) else ""
    return stderr or f"exit code {error.returncode}"


def _run_ffmpeg(*args: str) -> None:
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *args]
    try:
        subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True,
        )
    except FileNotFoundError as error:
        raise ValueError("ffmpeg not found; install ffmpeg and try again") from error
    except subprocess.CalledProcessError as error:
        raise ValueError(f"ffmpeg conversion failed: {_error_detail(error)}") from error


def _duration(path: Path) -> float:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=nw=1:nk=1",
        str(path),
    ]
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True,
        )
    except FileNotFoundError as error:
        raise ValueError("ffprobe not found; install ffmpeg and try again") from error
    except subprocess.CalledProcessError as error:
        raise ValueError(f"ffprobe failed: {_error_detail(error)}") from error

    try:
        duration = float(result.stdout.strip())
    except ValueError as error:
        raise ValueError("ffprobe returned an invalid duration") from error
    if not np.isfinite(duration) or duration < 0:
        raise ValueError("ffprobe returned an invalid duration")
    return duration


def find_silences(path: Path) -> list[float]:
    command = [
        "ffmpeg",
        "-hide_banner",
        "-i",
        str(path),
        "-af",
        f"silencedetect=noise={SILENCE_DB}dB:d={SILENCE_MIN}",
        "-f",
        "null",
        "-",
    ]
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True,
        )
    except FileNotFoundError as error:
        raise ValueError("ffmpeg not found; install ffmpeg and try again") from error
    except subprocess.CalledProcessError as error:
        raise ValueError(
            f"ffmpeg silencedetect failed: {_error_detail(error)}"
        ) from error

    points: list[float] = []
    start: float | None = None
    for line in result.stderr.splitlines():
        if "silence_start:" in line:
            start = float(line.split("silence_start:")[1].strip())
        elif "silence_end:" in line and start is not None:
            end = float(line.split("silence_end:")[1].split("|")[0].strip())
            points.append((start + end) / 2)
            start = None
    return points


def chunk_sample_bounds(
    total_samples: int, silence_samples: list[int]
) -> list[tuple[int, int]]:
    if total_samples < 0:
        raise ValueError("total sample count must be non-negative")

    silences = sorted(
        {sample for sample in silence_samples if 0 < sample < total_samples}
    )
    bounds: list[tuple[int, int]] = []
    position = 0
    while total_samples - position > MAX_CHUNK_SAMPLES:
        limit = position + MAX_CHUNK_SAMPLES
        candidates = [
            sample
            for sample in silences
            if position + MIN_PAUSE_OFFSET_SAMPLES < sample <= limit
        ]
        cut = candidates[-1] if candidates else limit
        bounds.append((position, cut))
        position = cut
    bounds.append((position, total_samples))
    return bounds


def chunk_bounds(total: float, silences: list[float]) -> list[tuple[float, float]]:
    if not np.isfinite(total) or total < 0:
        raise ValueError("total duration must be a non-negative finite number")
    total_samples = round(total * SAMPLE_RATE)
    silence_samples = [round(point * SAMPLE_RATE) for point in silences]
    return [
        (start / SAMPLE_RATE, end / SAMPLE_RATE)
        for start, end in chunk_sample_bounds(total_samples, silence_samples)
    ]


def _validate_wav(wav_file: wave.Wave_read, path: Path) -> int:
    if wav_file.getsampwidth() != 2 or wav_file.getnchannels() != 1:
        raise ValueError(f"expected 16-bit mono wav: {path}")
    if wav_file.getframerate() != SAMPLE_RATE:
        raise ValueError(f"expected {SAMPLE_RATE} Hz wav: {path}")
    return wav_file.getnframes()


def _wav_frame_count(path: Path) -> int:
    try:
        with wave.open(str(path), "rb") as wav_file:
            return _validate_wav(wav_file, path)
    except (EOFError, OSError, wave.Error) as error:
        raise ValueError(f"invalid converted wav: {path}") from error


def _read_wav(path: Path, start: int = 0, end: int | None = None) -> np.ndarray:
    try:
        with wave.open(str(path), "rb") as wav_file:
            total = _validate_wav(wav_file, path)
            stop = total if end is None else end
            if not 0 <= start <= stop <= total:
                raise ValueError(
                    f"invalid wav sample range [{start}, {stop}) for {total} samples"
                )
            wav_file.setpos(start)
            data = wav_file.readframes(stop - start)
    except (EOFError, OSError, wave.Error) as error:
        raise ValueError(f"invalid converted wav: {path}") from error

    samples = np.frombuffer(data, dtype=np.int16)
    if len(samples) != stop - start:
        raise ValueError(f"incomplete wav sample range from {path}")
    return samples.astype(np.float32) / 32768.0


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


def find_model_dir(explicit: Path | None = None) -> Path:
    """Compatibility alias for path selection; loading validates the selected bundle."""
    return resolve_model_dir(explicit)


class GigaAMEngine:
    def __init__(
        self, model_dir: Path | None = None, threads: int = 0,
        *, loaded_model: LoadedModel | None = None,
    ) -> None:
        if loaded_model is not None and (model_dir is not None or threads != 0):
            raise ValueError("Pass either loaded_model or model_dir/threads, not both")
        model = (
            loaded_model if loaded_model is not None else load_model(model_dir, threads)
        )
        with model:
            self.model_dir = model.metadata.model_dir
            self.model_identity = model.metadata.model_identity()
            self.pred_hidden = int(model.config["head"]["decoder"]["pred_hidden"])
            self.pred_layers = int(model.config["head"]["decoder"]["pred_rnn_layers"])
            self.features = Features(model.config)
            self.blank_id = len(model.tokenizer)
            (
                self.encoder, self.decoder, self.joint, self.tokenizer
            ) = model.take_runtime()

    def transcribe_wave(self, wav: np.ndarray) -> str:
        features = self.features(wav)
        frame_count = self.features.out_len(len(wav))
        if frame_count == 0 or features.shape[2] == 0:
            return ""
        lengths = np.array([frame_count], dtype=np.int64)
        outputs = self.encoder.run(
            ["encoded", "encoded_len"],
            {"audio_signal": features, "length": lengths},
        )
        encoded = np.asarray(outputs[0], dtype=np.float32)
        encoded_length = int(np.asarray(outputs[1]).reshape(-1)[0])
        return self.tokenizer.decode(self._decode(encoded, encoded_length))

    def _decode(self, encoded: np.ndarray, encoded_length: int) -> list[int]:
        dtype = np.float32
        encoded = np.asarray(encoded, dtype=dtype, order="C")

        hypothesis: list[int] = []
        label = np.array([[self.blank_id]], dtype=np.int64)
        hidden = np.zeros((self.pred_layers, 1, self.pred_hidden), dtype=dtype)
        cell = np.zeros_like(hidden)
        started = False

        for frame_index in range(min(encoded_length, encoded.shape[2])):
            frame = encoded[:, :, frame_index : frame_index + 1]
            for _ in range(MAX_SYMBOLS_PER_FRAME):
                if started:
                    args = [label, hidden, cell]
                else:
                    args = [
                        np.array([[self.blank_id]], dtype=np.int64),
                        np.zeros_like(hidden),
                        np.zeros_like(cell),
                    ]
                prediction, next_hidden, next_cell = self.decoder.run(
                    ["dec", "ho", "co"],
                    {"x": args[0], "hi": args[1], "ci": args[2]},
                )
                prediction = np.asarray(prediction, dtype=dtype)
                output = self.joint.run(
                    ["joint"],
                    {"enc": frame, "dec": prediction.swapaxes(1, 2)},
                )
                logits = np.asarray(output[0])
                token = int(logits[:, 0, 0, :].argmax(axis=-1)[0])
                if token == self.blank_id:
                    break
                hypothesis.append(token)
                label = np.array([[token]], dtype=np.int64)
                hidden = np.asarray(next_hidden, dtype=dtype)
                cell = np.asarray(next_cell, dtype=dtype)
                started = True
        return hypothesis

    def transcribe(self, input_path: Path) -> list[GigaAMSegment]:
        """Compatibility API for callers that only need speech chunks."""
        result = self.transcribe_result(input_path)
        return [GigaAMSegment(**segment) for segment in result.raw_segments]

    def transcribe_result(self, input_path: Path) -> TranscriptResult:
        """Return speech chunks and the full normalized WAV duration."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            wav_path = root / "input.wav"
            _run_ffmpeg(
                "-i",
                str(input_path),
                "-ac",
                "1",
                "-ar",
                str(SAMPLE_RATE),
                str(wav_path),
            )
            _duration(wav_path)
            total_samples = _wav_frame_count(wav_path)
            silence_samples = (
                []
                if total_samples <= MAX_CHUNK_SAMPLES
                else [
                    round(point * SAMPLE_RATE) for point in find_silences(wav_path)
                ]
            )
            bounds = chunk_sample_bounds(total_samples, silence_samples)
            segments: list[GigaAMSegment] = []
            for start_sample, end_sample in bounds:
                if end_sample <= start_sample:
                    continue
                text = self.transcribe_wave(
                    _read_wav(wav_path, start_sample, end_sample)
                ).strip()
                if text:
                    segments.append(
                        GigaAMSegment(
                            start=start_sample / SAMPLE_RATE,
                            end=end_sample / SAMPLE_RATE,
                            text=text,
                        )
                    )
            return TranscriptResult(
                raw_segments=[
                    {"start": item.start, "end": item.end, "text": item.text}
                    for item in segments
                ],
                language="ru",
                duration_seconds=total_samples / SAMPLE_RATE,
            )
