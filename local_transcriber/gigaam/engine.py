"""Local GigaAM model loading and RNNT inference."""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from local_transcriber.gigaam.audio import (
    MAX_CHUNK_SAMPLES,
    SAMPLE_RATE,
    _duration,
    _read_wav,
    _run_ffmpeg,
    _wav_frame_count,
    chunk_sample_bounds,
    find_silences,
)
from local_transcriber.gigaam.features import Features
from local_transcriber.models.validation import (
    BUNDLE,
    LoadedModel,
    load_model,
    resolve_model_dir,
)
from local_transcriber.transcript import TranscriptResult

MODEL_NAME = BUNDLE.profile
MAX_SYMBOLS_PER_FRAME = 3


@dataclass(frozen=True)
class GigaAMSegment:
    start: float
    end: float
    text: str


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
