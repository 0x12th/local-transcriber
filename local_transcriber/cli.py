from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
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
        description=(
            "Transcribe audio or video locally with Whisper and write plain text, "
            "timestamped Markdown, and JSON."
        ),
    )
    parser.add_argument("input", type=Path, help="Audio or video file to transcribe.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--whisper-model", default="turbo")
    parser.add_argument(
        "--language",
        default=None,
        help="Language code, for example ru or en. Omit for auto-detection.",
    )
    parser.add_argument(
        "--speaker-count",
        type=positive_int,
        default=None,
        help=(
            "Expected number of speakers. Stored as metadata; Whisper does not "
            "diarize speakers."
        ),
    )
    parser.add_argument(
        "--initial-prompt",
        default=None,
        help="Domain terms, names, acronyms, or style hints for Whisper.",
    )
    parser.add_argument(
        "--prompt-speakers",
        action="store_true",
        help="Append the expected speaker count to the initial prompt.",
    )
    parser.add_argument(
        "--merge-gap-seconds",
        type=non_negative_float,
        default=1.5,
        help="Merge adjacent segments separated by at most this many seconds.",
    )
    parser.add_argument(
        "--min-segment-seconds",
        type=non_negative_float,
        default=0.0,
        help="Drop shorter segments; useful for noisy recordings.",
    )
    parser.add_argument(
        "--drop-subtitle-artifacts",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Drop obvious subtitle or caption credit artifacts.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cpu", "cuda", "mps"),
        help="Compute device. Auto selects CUDA, then MPS, then CPU.",
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def choose_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def build_initial_prompt(
    initial_prompt: str | None,
    speaker_count: int | None,
    prompt_speakers: bool,
) -> str | None:
    prompt_parts = []
    if initial_prompt:
        prompt_parts.append(initial_prompt.strip())
    if prompt_speakers and speaker_count is not None:
        prompt_parts.append(f"The recording contains {speaker_count} speakers.")
    return " ".join(part for part in prompt_parts if part) or None


def transcribe(
    input_path: Path,
    model_name: str,
    language: str | None,
    device: str,
    initial_prompt: str | None,
) -> dict[str, Any]:
    model = whisper.load_model(model_name, device=device)
    options: dict[str, Any] = {
        "task": "transcribe",
        "verbose": False,
        "fp16": device == "cuda",
    }
    if language:
        options["language"] = language
    if initial_prompt:
        options["initial_prompt"] = initial_prompt
    return model.transcribe(str(input_path), **options)


def maybe_float(value: Any) -> float | None:
    return None if value is None else float(value)


def build_segments(result: dict[str, Any]) -> list[TranscriptSegment]:
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
    segments: list[TranscriptSegment],
    max_gap_seconds: float,
) -> list[TranscriptSegment]:
    if not segments:
        return []
    merged = [segments[0]]
    for segment in segments[1:]:
        previous = merged[-1]
        if segment.start - previous.end <= max_gap_seconds:
            merged[-1] = TranscriptSegment(
                previous.start,
                segment.end,
                f"{previous.text} {segment.text}",
            )
        else:
            merged.append(segment)
    return merged


def format_timestamp(seconds: float) -> str:
    total = round(seconds)
    hours, remainder = divmod(total, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def write_outputs(
    out_dir: Path,
    input_path: Path,
    segments: list[TranscriptSegment],
    result: dict[str, Any],
    args: argparse.Namespace,
    initial_prompt: str | None,
    device: str,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp_lines = ["# Transcript with Timestamps", ""]
    for segment in segments:
        time_range = (
            f"{format_timestamp(segment.start)}-{format_timestamp(segment.end)}"
        )
        timestamp_lines.extend(
            [
                f"**{time_range}:** {segment.text}",
                "",
            ]
        )
    transcript_lines = ["# Transcript", "", *(segment.text for segment in segments), ""]
    metadata = {
        "input": str(input_path),
        "language": result.get("language"),
        "requested_language": args.language,
        "whisper_model": args.whisper_model,
        "device": device,
        "requested_device": args.device,
        "speaker_count": args.speaker_count,
        "initial_prompt": initial_prompt,
        "segments": [asdict(segment) for segment in segments],
    }
    (out_dir / "transcript_timestamps.md").write_text(
        "\n".join(timestamp_lines), encoding="utf-8"
    )
    (out_dir / "transcript.md").write_text(
        "\n".join(transcript_lines), encoding="utf-8"
    )
    (out_dir / "transcript_data.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    input_path = args.input.expanduser().resolve()
    if not input_path.is_file():
        build_parser().error(f"input file not found: {input_path}")

    device = choose_device(args.device)
    initial_prompt = build_initial_prompt(
        args.initial_prompt, args.speaker_count, args.prompt_speakers
    )
    language_label = args.language or "auto"
    configuration = f"{args.whisper_model}, {device}, {language_label}"
    print(f"Transcribing with Whisper ({configuration})...")
    result = transcribe(
        input_path, args.whisper_model, args.language, device, initial_prompt
    )
    segments = clean_segments(
        build_segments(result),
        args.min_segment_seconds,
        args.drop_subtitle_artifacts,
    )
    segments = merge_adjacent_segments(segments, args.merge_gap_seconds)
    write_outputs(
        args.out_dir, input_path, segments, result, args, initial_prompt, device
    )
    print(f"Done: {args.out_dir / 'transcript_timestamps.md'}")
    print(f"Done: {args.out_dir / 'transcript.md'}")
    print(f"Data: {args.out_dir / 'transcript_data.json'}")
