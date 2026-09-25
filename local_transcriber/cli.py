from __future__ import annotations

import argparse
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

from local_transcriber.outputs import (
    create_run_directory as create_run_directory,
)
from local_transcriber.outputs import (
    format_timestamp as format_timestamp,
)
from local_transcriber.outputs import (
    output_paths,
    serialize_outputs,
)
from local_transcriber.outputs import write_outputs as write_serialized_outputs
from local_transcriber.transcript import (
    SUBTITLE_ARTIFACT_RE as SUBTITLE_ARTIFACT_RE,
)
from local_transcriber.transcript import (
    ProcessingPolicy,
    RunMetadata,
    TranscriptResult,
    TranscriptSegment,
    build_segments,
    process_transcript,
)
from local_transcriber.transcript import clean_segments as clean_segments
from local_transcriber.transcript import maybe_float as maybe_float
from local_transcriber.transcript import (
    merge_adjacent_segments as merge_adjacent_segments,
)

DEFAULT_OUT_DIR = Path("out")


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
            "Transcribe audio or video locally with Whisper or GigaAM and write "
            "plain text, timestamped Markdown, and JSON."
        ),
    )
    parser.add_argument("input", type=Path, help="Audio or video file to transcribe.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--engine",
        choices=("whisper", "gigaam"),
        default="whisper",
        help="Transcription engine. Whisper remains the default.",
    )
    parser.add_argument(
        "--gigaam-model-dir",
        type=Path,
        default=None,
        help="GigaAM model directory; no model is downloaded during transcription.",
    )

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
        help=(
            "Merge adjacent Whisper segments separated by at most this many seconds; "
            "GigaAM chunks remain separate."
        ),
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
        default=False,
        help="Drop subtitle/credit keyword matches; may remove real speech (opt-in).",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cpu", "cuda", "mps"),
        help=(
            "Compute device. Whisper auto-selects CUDA, then MPS, then CPU; "
            "GigaAM accepts auto or cpu and runs on CPU."
        ),
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def choose_device(requested: str) -> str:
    import torch

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
    import whisper

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


def validate_gigaam_options(args: argparse.Namespace) -> None:
    if args.language not in (None, "ru"):
        raise ValueError("GigaAM supports only --language ru or automatic Russian")
    if args.device not in ("auto", "cpu"):
        raise ValueError("GigaAM runs on CPU; use --device auto or --device cpu")
    if args.initial_prompt is not None or args.prompt_speakers:
        raise ValueError(
            "GigaAM does not support --initial-prompt or --prompt-speakers"
        )


def transcribe_gigaam(
    input_path: Path, model_dir: Path | None
) -> tuple[list[TranscriptSegment], dict[str, Any], Path, str]:
    try:
        from local_transcriber.gigaam import MODEL_NAME, GigaAMEngine

        engine = GigaAMEngine(model_dir=model_dir)
    except ImportError as error:
        raise ValueError(
            "GigaAM dependencies could not be imported; run "
            "`uv sync --locked --extra gigaam`"
        ) from error

    raw = engine.transcribe_result(input_path)
    result = {
        "language": raw.language,
        "segments": raw.raw_segments,
        "duration_seconds": raw.duration_seconds,
        "model_identity": engine.model_identity,
    }
    return build_segments(result), result, engine.model_dir, MODEL_NAME


def processing_policy(args: argparse.Namespace) -> ProcessingPolicy:
    return ProcessingPolicy(
        min_segment_seconds=args.min_segment_seconds,
        drop_subtitle_artifacts=args.drop_subtitle_artifacts,
        merge_gap_seconds=args.merge_gap_seconds,
        merge_policy="none" if args.engine == "gigaam" else "adjacent",
    )


def run_metadata(
    input_path: Path,
    args: argparse.Namespace,
    initial_prompt: str | None,
    device: str,
    started_at: datetime,
    gigaam_model_dir: Path | None = None,
    gigaam_profile: str | None = None,
    model_identity: dict[str, Any] | None = None,
) -> RunMetadata:
    return RunMetadata(
        input_path=input_path,
        started_at=started_at,
        engine=args.engine,
        requested_language=args.language,
        whisper_model=args.whisper_model if args.engine == "whisper" else None,
        gigaam_profile=gigaam_profile,
        gigaam_model_dir=gigaam_model_dir,
        device=device,
        requested_device=args.device,
        speaker_count=args.speaker_count,
        initial_prompt=initial_prompt,
        model_identity=model_identity if args.engine == "gigaam" else None,
    )


def write_outputs(
    out_dir: Path,
    input_path: Path,
    segments: list[TranscriptSegment],
    result: dict[str, Any],
    args: argparse.Namespace,
    initial_prompt: str | None,
    device: str,
    started_at: datetime,
    gigaam_model_dir: Path | None = None,
    gigaam_profile: str | None = None,
) -> None:
    """Legacy CLI adapter; derive provenance from RAW, never from text matching."""
    raw = TranscriptResult.from_engine_result(result)
    view = process_transcript(raw, processing_policy(args))
    if list(view.segments) != segments:
        raise ValueError(
            "segments do not match RAW and the requested processing policy"
        )
    metadata = run_metadata(
        input_path,
        args,
        initial_prompt,
        device,
        started_at,
        gigaam_model_dir,
        gigaam_profile,
        result.get("model_identity"),
    )
    write_serialized_outputs(out_dir, serialize_outputs(raw, view, metadata))


def main(argv: list[str] | None = None) -> None:
    started_at = datetime.now().astimezone()
    parser = build_parser()
    args = parser.parse_args(argv)
    input_path = args.input.expanduser().resolve()
    if not input_path.is_file():
        parser.error(f"input file not found: {input_path}")

    args.out_dir = args.out_dir.expanduser().resolve()
    try:
        if args.engine == "gigaam":
            validate_gigaam_options(args)
            device = "cpu"
        else:
            device = choose_device(args.device)
        run_dir = create_run_directory(args.out_dir, input_path, started_at)
    except (OSError, ValueError) as error:
        parser.error(str(error))

    initial_prompt: str | None = None
    gigaam_model_dir: Path | None = None
    gigaam_profile: str | None = None
    try:
        if args.engine == "gigaam":
            print("Transcribing with GigaAM v3 (ONNX, CPU)...")
            _, result, gigaam_model_dir, gigaam_profile = transcribe_gigaam(
                input_path, args.gigaam_model_dir
            )
        else:
            initial_prompt = build_initial_prompt(
                args.initial_prompt, args.speaker_count, args.prompt_speakers
            )
            language_label = args.language or "auto"
            configuration = f"{args.whisper_model}, {device}, {language_label}"
            print(f"Transcribing with Whisper ({configuration})...")
            result = transcribe(
                input_path,
                args.whisper_model,
                args.language,
                device,
                initial_prompt,
            )

        raw = TranscriptResult.from_engine_result(result)
        view = process_transcript(raw, processing_policy(args))
        metadata = run_metadata(
            input_path,
            args,
            initial_prompt,
            device,
            started_at,
            gigaam_model_dir,
            gigaam_profile,
            result.get("model_identity"),
        )
        write_serialized_outputs(run_dir, serialize_outputs(raw, view, metadata))
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        parser.error(str(error))

    paths = output_paths(run_dir)
    print(f"Done: {paths[0]}")
    print(f"Done: {paths[1]}")
    print(f"Data: {paths[2]}")
