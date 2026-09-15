from __future__ import annotations

import os
import subprocess
import tempfile
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16_000
MODEL_NAME = "v3_e2e_rnnt"
MAX_CHUNK = 24.0
SILENCE_DB = -35
SILENCE_MIN = 0.3
MAX_SYMBOLS_PER_FRAME = 3


@dataclass(frozen=True)
class GigaAMSegment:
    start: float
    end: float
    text: str


def _run_ffmpeg(*args: str) -> None:
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *args],
        check=True,
    )


def _duration(path: Path) -> float:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=nw=1:nk=1",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return float(result.stdout.strip())


def find_silences(path: Path) -> list[float]:
    result = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-i",
            str(path),
            "-af",
            f"silencedetect=noise={SILENCE_DB}dB:d={SILENCE_MIN}",
            "-f",
            "null",
            "-",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
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


def chunk_bounds(total: float, silences: list[float]) -> list[tuple[float, float]]:
    bounds: list[tuple[float, float]] = []
    position = 0.0
    while total - position > MAX_CHUNK:
        candidates = [
            point
            for point in silences
            if position + 3 < point <= position + MAX_CHUNK
        ]
        cut = candidates[-1] if candidates else position + MAX_CHUNK
        bounds.append((position, cut))
        position = cut
    bounds.append((position, total))
    return bounds


def _read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as wav_file:
        if wav_file.getsampwidth() != 2 or wav_file.getnchannels() != 1:
            raise ValueError(f"expected 16-bit mono wav: {path}")
        data = wav_file.readframes(wav_file.getnframes())
    return np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0


def _hz_to_mel(freq: np.ndarray | float) -> np.ndarray | float:
    return 2595.0 * np.log10(1.0 + freq / 700.0)


def _mel_to_hz(mels: np.ndarray | float) -> np.ndarray | float:
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
    def __init__(self, config: dict) -> None:
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
            return samples // self.hop_length + 1
        return (samples - self.win_length) // self.hop_length + 1

    def __call__(self, wav: np.ndarray) -> np.ndarray:
        audio = np.asarray(wav, dtype=np.float32)
        if self.center:
            pad = self.n_fft // 2
            audio = np.pad(audio, (pad, pad), mode="reflect")
        frame_count = max(0, (len(audio) - self.n_fft) // self.hop_length + 1)
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
    candidates = [
        explicit,
        Path(os.environ["GIGAAM_MODEL_DIR"]) if "GIGAAM_MODEL_DIR" in os.environ else None,
        Path.home() / ".giga" / "model",
    ]
    for candidate in candidates:
        if candidate and (candidate / f"{MODEL_NAME}.yaml").exists():
            return candidate
    raise FileNotFoundError(
        "GigaAM model not found; pass --gigaam-model-dir or set GIGAAM_MODEL_DIR"
    )


class GigaAMEngine:
    def __init__(self, model_dir: Path | None = None, threads: int = 0) -> None:
        import onnxruntime as ort
        import yaml
        from sentencepiece import SentencePieceProcessor

        self.model_dir = find_model_dir(model_dir)
        with (self.model_dir / f"{MODEL_NAME}.yaml").open(encoding="utf-8") as file:
            config = yaml.safe_load(file)

        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.intra_op_num_threads = threads or min(8, os.cpu_count() or 4)
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.log_severity_level = 3

        def session(suffix: str):
            return ort.InferenceSession(
                str(self.model_dir / f"{MODEL_NAME}_{suffix}.onnx"),
                providers=["CPUExecutionProvider"],
                sess_options=options,
            )

        self.encoder = session("encoder")
        self.decoder = session("decoder")
        self.joint = session("joint")
        tokenizer_path = self.model_dir / f"{MODEL_NAME}_tokenizer.model"
        self.tokenizer = SentencePieceProcessor()
        self.tokenizer.load(str(tokenizer_path))
        self.blank_id = len(self.tokenizer)
        self.pred_hidden = int(config["head"]["decoder"]["pred_hidden"])
        self.pred_layers = int(config["head"]["decoder"]["pred_rnn_layers"])
        self.features = Features(config)

    def transcribe_wave(self, wav: np.ndarray) -> str:
        features = self.features(wav)
        lengths = np.array([self.features.out_len(len(wav))], dtype=np.int64)
        outputs = self.encoder.run(
            [item.name for item in self.encoder.get_outputs()],
            {
                item.name: value
                for item, value in zip(
                    self.encoder.get_inputs(), [features, lengths], strict=True
                )
            },
        )
        encoded, encoded_length = outputs[0], int(np.asarray(outputs[1]).reshape(-1)[0])
        return self.tokenizer.decode(self._decode(encoded, encoded_length))

    def _decode(self, encoded: np.ndarray, encoded_length: int) -> list[int]:
        dtype = np.float32
        encoded = np.asarray(encoded, dtype=dtype, order="C")
        decoder_outputs = [item.name for item in self.decoder.get_outputs()]
        joint_outputs = [item.name for item in self.joint.get_outputs()]
        decoder_inputs = [item.name for item in self.decoder.get_inputs()]
        joint_inputs = [item.name for item in self.joint.get_inputs()]
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
                    decoder_outputs, dict(zip(decoder_inputs, args, strict=True))
                )
                output = self.joint.run(
                    joint_outputs,
                    dict(
                        zip(
                            joint_inputs,
                            [frame, prediction.swapaxes(1, 2)],
                            strict=True,
                        )
                    ),
                )
                token = int(output[0][:, 0, 0, :].argmax(axis=-1)[0])
                if token == self.blank_id:
                    break
                hypothesis.append(token)
                label = np.array([[token]], dtype=np.int64)
                hidden, cell, started = next_hidden, next_cell, True
        return hypothesis

    def transcribe(self, input_path: Path) -> list[GigaAMSegment]:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            wav_path = root / "input.wav"
            _run_ffmpeg(
                "-i", str(input_path), "-ac", "1", "-ar", str(SAMPLE_RATE), str(wav_path)
            )
            total = _duration(wav_path)
            bounds = (
                [(0.0, total)]
                if total <= MAX_CHUNK + 1
                else chunk_bounds(total, find_silences(wav_path))
            )
            segments: list[GigaAMSegment] = []
            for index, (start, end) in enumerate(bounds):
                part = root / f"part-{index}.wav"
                if len(bounds) == 1:
                    part = wav_path
                else:
                    _run_ffmpeg(
                        "-i",
                        str(wav_path),
                        "-ss",
                        str(start),
                        "-to",
                        str(end),
                        str(part),
                    )
                text = self.transcribe_wave(_read_wav(part)).strip()
                if text:
                    segments.append(GigaAMSegment(start=start, end=end, text=text))
            return segments
