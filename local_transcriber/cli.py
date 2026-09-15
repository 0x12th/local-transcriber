from __future__ import annotations

import argparse
import json
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import whisper

DEFAULT_OUT_DIR = Path("out")
SUBTITLE_ARTIFACT_RE = re.compile(
    r"(редактор субтитров|корректор|субтитры|subtitles?|caption|captions?|"
    r"created by|transcribed by)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class TranscriptSegment:
    start: float
    end: float
    text: str
    avg_logprob: float | None = None
    no_speech_prob: float | None = None


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return parsed


def non_negative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("expected a non-negative number")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="local-transcriber",
        description="Transcribe audio or video locally with Whisper or GigaAM.",
    )
    parser.add_argument("input", type=Path, help="Audio or video file to transcribe.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--engine", choices=("whisper", "gigaam"), default="whisper")
    parser.add_argument("--whisper-model", default="turbo")
    parser.add_argument("--gigaam-model-dir", type=Path, default=None)
    parser.add_argument("--language", default=None)
    parser.add_argument("--speaker-count", type=positive_int, default=None)
    parser.add_argument("--initial-prompt", default=None)
    parser.add_argument("--prompt-speakers", action="store_true")
    parser.add_argument("--merge-gap-seconds", type=non_negative_float, default=1.5)
    parser.add_argument("--min-segment-seconds", type=non_negative_float, default=0.0)
    parser.add_argument(
        "--drop-subtitle-artifacts",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cpu", "cuda", "mps"),
        help="Whisper device. GigaAM prototype currently uses CPU ONNX.",
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def choose_device(requested: str) -> str:
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is not available; use --device cpu or --device auto")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS is not available; use --device cpu or --device auto")
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def build_initial_prompt(
    initial_prompt: str | None, speaker_count: int | None, prompt_speakers: bool
) -> str | None:
    parts = []
    if initial_prompt:
        parts.append(initial_prompt.strip())
    if prompt_speakers and speaker_count is not None:
        parts.append(f"The recording contains {speaker_count} speakers.")
    return " ".join(part for part in parts if part) or None


def transcribe_whisper(
    input_path: Path,
    model_name: str,
    language: str | None,
    device: str,
    initial_prompt: str | None,
) -> dict[str, Any]:
    model = whisper.load_model(model_name, device=device)
    options: dict[str, Any] = {"task": "transcribe", "verbose": False, "fp16": device == "cuda"}
    if language:
        options["language"] = language
    if initial_prompt:
        options["initial_prompt"] = initial_prompt
    return model.transcribe(str(input_path), **options)


def maybe_float(value: Any) -> float | None:
    return None if value is None else float(value)


def build_whisper_segments(result: dict[str, Any]) -> list[TranscriptSegment]:
    segments = []
    for segment in result["segments"]:
        text = str(segment["text"]).strip()
        if text:
            segments.append(
                TranscriptSegment(
                    start=float(segment["start"]),
                    end=float(segment["end"]),
                    text=text,
                    avg_logprob=maybe_float(segment.get("avg_logprob")),
                    no_speech_prob=maybe_float(segment.get("no_speech_prob")),
                )
            )
    return segments


def transcribe_gigaam(
    input_path: Path, model_dir: Path | None
) -> tuple[list[TranscriptSegment], dict[str, Any]]:
    from local_transcriber.gigaam import GigaAMEngine

    raw = GigaAMEngine(model_dir=model_dir).transcribe(input_path)
    segments = [TranscriptSegment(item.start, item.end, item.text) for item in raw]
    result = {
        "language": "ru",
        "segments": [
            {"start": item.start, "end": item.end, "text": item.text} for item in raw
        ],
    }
    return segments, result


def clean_segments(
    segments: list[TranscriptSegment],
    min_segment_seconds: float,
    drop_subtitle_artifacts: bool,
) -> list[TranscriptSegment]:
    return [
        segment
        for segment in segments
        if segment.end - segment.start >= min_segment_seconds
        and not (drop_subtitle_artifacts and SUBTITLE_ARTIFACT_RE.search(segment.text))
    ]


def merge_adjacent_segments(
    segments: list[TranscriptSegment], max_gap_seconds: float
) -> list[TranscriptSegment]:
    if not segments:
        return []
    merged = [segments[0]]
    for segment in segments[1:]:
        previous = merged[-1]
        if segment.start - previous.end <= max_gap_seconds:
            merged[-1] = TranscriptSegment(
                previous.start, segment.end, f"{previous.text} {segment.text}"
            )
        else:
            merged.append(segment)
    return merged


def format_timestamp(seconds: float) -> str:
    total = round(seconds)
    hours, remainder = divmod(total, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def output_paths(out_dir: Path) -> tuple[Path, Path, Path]:
    return (
        out_dir / "transcript_timestamps.md",
        out_dir / "transcript.md",
        out_dir / "transcript.json",
    )


def create_run_directory(out_dir: Path, input_path: Path, started_at: datetime) -> Path:
    parent = out_dir / input_path.stem
    parent.mkdir(parents=True, exist_ok=True)
    name = started_at.strftime("%Y-%m-%d_%H-%M-%S")
    attempt = 1
    while True:
        candidate = parent / (name if attempt == 1 else f"{name}-{attempt}")
        try:
            candidate.mkdir(exist_ok=False)
        except FileExistsError:
            attempt += 1
        else:
            return candidate


def write_outputs(
    out_dir: Path,
    input_path: Path,
    segments: list[TranscriptSegment],
    result: dict[str, Any],
    args: argparse.Namespace,
    initial_prompt: str | None,
    device: str,
    started_at: datetime,
) -> None:
    paths = output_paths(out_dir)
    timestamp_lines = ["# Transcript with Timestamps", ""]
    for segment in segments:
        time_range = f"{format_timestamp(segment.start)}-{format_timestamp(segment.end)}"
        timestamp_lines.extend([f"**{time_range}:** {segment.text}", ""])
    transcript_lines = ["# Transcript", "", *(segment.text for segment in segments), ""]
    metadata = {
        "input": str(input_path),
        "started_at": started_at.isoformat(),
        "engine": args.engine,
        "language": result.get("language"),
        "requested_language": args.language,
        "whisper_model": args.whisper_model if args.engine == "whisper" else None,
        "gigaam_model_dir": str(args.gigaam_model_dir) if args.gigaam_model_dir else None,
        "device": device,
        "speaker_count": args.speaker_count,
        "initial_prompt": initial_prompt,
        "raw_segments": result["segments"],
        "merged_segments": [
            {"start": segment.start, "end": segment.end, "text": segment.text}
            for segment in segments
        ],
    }
    contents = (
        "\n".join(timestamp_lines),
        "\n".join(transcript_lines),
        json.dumps(metadata, ensure_ascii=False, indent=2),
    )
    for path, content in zip(paths, contents, strict=True):
        with path.open("x", encoding="utf-8") as output:
            output.write(content)


def main(argv: list[str] | None = None) -> None:
    started_at = datetime.now().astimezone()
    args = parse_args(argv)
    input_path = args.input.expanduser().resolve()
    if not input_path.is_file():
        build_parser().error(f"input file not found: {input_path}")
    args.out_dir = args.out_dir.expanduser().resolve()
    initial_prompt = build_initial_prompt(
        args.initial_prompt, args.speaker_count, args.prompt_speakers
    )
    try:
        run_dir = create_run_directory(args.out_dir, input_path, started_at)
        if args.engine == "gigaam":
            if args.language not in (None, "ru"):
                raise ValueError("GigaAM backend currently supports Russian only")
            device = "cpu"
            print("Transcribing with GigaAM v3 (ONNX, CPU)...")
            segments, result = transcribe_gigaam(input_path, args.gigaam_model_dir)
        else:
            device = choose_device(args.device)
            language_label = args.language or "auto"
            print(
                f"Transcribing with Whisper ({args.whisper_model}, "
                f"{device}, {language_label})..."
            )
            result = transcribe_whisper(
                input_path, args.whisper_model, args.language, device, initial_prompt
            )
            segments = build_whisper_segments(result)
        segments = clean_segments(
            segments, args.min_segment_seconds, args.drop_subtitle_artifacts
        )
        segments = merge_adjacent_segments(segments, args.merge_gap_seconds)
        write_outputs(
            run_dir, input_path, segments, result, args, initial_prompt, device, started_at
        )
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        build_parser().error(str(error))
    paths = output_paths(run_dir)
    print(f"Done: {paths[0]}")
    print(f"Done: {paths[1]}")
    print(f"Data: {paths[2]}")
