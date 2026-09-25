"""Audio conversion and chunking for the local GigaAM backend."""

from __future__ import annotations

import subprocess
import wave
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16_000
MAX_CHUNK = 24.0
MAX_CHUNK_SAMPLES = int(MAX_CHUNK * SAMPLE_RATE)
MIN_PAUSE_OFFSET_SAMPLES = 3 * SAMPLE_RATE
SILENCE_DB = -35
SILENCE_MIN = 0.3


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
