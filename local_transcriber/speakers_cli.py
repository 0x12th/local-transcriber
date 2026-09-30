"""Shared Whisper word alignment and offline Nemotron speaker processing."""

from __future__ import annotations

import gc
import json
import math
import subprocess
import tempfile
import wave
from contextlib import nullcontext
from pathlib import Path
from typing import Any
from unittest.mock import patch

from local_transcriber.diarization import (
    parse_rttm,
    speaker_items,
    speaker_markdown,
    speaker_segments,
)
from local_transcriber.speaker_postprocessing import postprocess_speakers


def _run(command: list[str], timeout: int) -> str:
    try:
        result = subprocess.run(
            command, capture_output=True, text=True, check=False, timeout=timeout,
        )
    except subprocess.TimeoutExpired as error:
        raise RuntimeError(
            f"{Path(command[0]).name} timed out after {timeout}s"
        ) from error
    if result.returncode:
        message = (result.stderr or result.stdout).strip()[-800:]
        raise RuntimeError(
            f"{Path(command[0]).name} exited {result.returncode}: {message}"
        )
    return result.stdout.strip()


def _wave_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as audio:
        if (audio.getnchannels(), audio.getsampwidth(), audio.getframerate()) != (
            1, 2, 16000
        ) or not audio.getnframes():
            raise ValueError("decoded audio is not nonempty mono 16 kHz PCM16")
        return audio.getnframes() / 16000


def _decode(source: Path, wav: Path) -> float:
    _run([
        "ffmpeg", "-nostdin", "-v", "error", "-i", str(source),
        "-map", "0:a:0", "-vn", "-af", "aresample=async=1:first_pts=0",
        "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", "-n", str(wav),
    ], timeout=3600)
    return _wave_duration(wav)


def _transcribe(
    wav: Path, checkpoint: str | Path, device: str,
    language: str | None, initial_prompt: str | None = None,
) -> dict[str, Any]:
    import torch
    import whisper
    import whisper.timing

    model = whisper.load_model(str(checkpoint), device=device)
    original_dtw = whisper.timing.dtw
    # This Whisper release moves MPS tensors to float64 before CPU DTW; MPS has
    # no float64. Copy only the alignment matrix to CPU first, not the ASR model.
    alignment = (
        patch.object(whisper.timing, "dtw", lambda x: original_dtw(x.cpu()))
        if device == "mps" else nullcontext()
    )
    try:
        with alignment:
            options: dict[str, Any] = {
                "task": "transcribe", "verbose": False,
                "word_timestamps": True, "fp16": device == "cuda",
            }
            if language:
                options["language"] = language
            if initial_prompt:
                options["initial_prompt"] = initial_prompt
            return model.transcribe(str(wav), **options)
    finally:
        del model
        gc.collect()
        if device == "mps":
            torch.mps.empty_cache()


def _diarize(executable: Path, model: Path, wav: Path, duration: float,
             device: str, timeout: int) -> tuple[str, list[dict[str, Any]]]:
    with tempfile.TemporaryDirectory(prefix="speaker-rttm-") as directory:
        destination = Path(directory) / "voices.rttm"
        _run([
            str(executable), "diarize", str(wav), "--model", str(model),
            "--device", device, "--format", "rttm", "--recording-id", "audio",
            "--output", str(destination),
        ], timeout=timeout)
        if not destination.is_file():
            raise ValueError("nemo-speech returned no RTTM file")
        text = destination.read_text(encoding="utf-8")
        return text, parse_rttm(text, "audio", duration)


def diarize_transcript(
    wav: Path, duration: float, raw_segments: list[dict[str, Any]],
    model: Path, executable: Path,
) -> tuple[bytes, bytes, str | None]:
    """Return speaker Markdown, compact turns JSON and an optional failure reason."""
    try:
        _, intervals = _diarize(executable, model, wav, duration, "metal", 3600)
        items = speaker_items(raw_segments, intervals)
        for item in items:
            if not (math.isfinite(item["start"]) and math.isfinite(item["end"])
                    and item["end"] <= duration + 0.05):
                raise ValueError("Whisper word timestamp exceeds audio duration")
        groups = postprocess_speakers(speaker_segments(items))
        error_message = None
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        error_message = str(error)
        # Keep raw text and unlabeled words if the diarizer or its RTTM fails.
        try:
            groups = speaker_segments(speaker_items(raw_segments, []))
        except (TypeError, ValueError):
            groups = []
    speaker_numbers: dict[str, int] = {}
    turns = []
    for group in groups:
        label = group["speaker"] if group["status"] == "assigned" else None
        if label is not None and label not in speaker_numbers:
            speaker_numbers[label] = len(speaker_numbers) + 1
        turns.append({
            "start": group["start"],
            "end": group["end"],
            "text": group["text"].strip(),
            "speaker": speaker_numbers[label] if label is not None else None,
        })
    return (
        speaker_markdown(groups, "failed" if error_message is not None else "success"),
        (json.dumps(turns, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        .encode("utf-8"),
        error_message,
    )


def _write_run(root: Path, artifacts: dict[str, bytes]) -> None:
    # The caller owns an exclusive run directory.
    for name, content in artifacts.items():
        with (root / name).open("xb") as output:
            output.write(content)
