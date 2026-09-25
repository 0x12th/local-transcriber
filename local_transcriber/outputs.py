"""Transcript serialization and exclusive file publication, without CLI or ASR."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from local_transcriber.transcript import RunMetadata, TranscriptResult, TranscriptView


@dataclass(frozen=True)
class SerializedOutputs:
    timestamped_markdown: bytes
    markdown: bytes
    transcript_json: bytes


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


def serialize_transcript(
    result: TranscriptResult, view: TranscriptView, metadata: RunMetadata
) -> bytes:
    """Return the exact UTF-8 JSON bytes to publish (and, later, hash)."""
    expected_merge = "none" if metadata.engine == "gigaam" else "adjacent"
    if view.processing.merge_policy != expected_merge:
        raise ValueError(f"{metadata.engine} requires merge_policy={expected_merge}")
    if metadata.engine == "gigaam" and result.duration_seconds is None:
        raise ValueError("GigaAM requires full normalized audio duration_seconds")
    if result.duration_seconds is not None and (
        not math.isfinite(result.duration_seconds) or result.duration_seconds < 0
    ):
        raise ValueError("duration_seconds must be finite and non-negative")
    for indices in view.view_raw_indices:
        if not indices or any(not 0 <= i < len(result.raw_segments) for i in indices):
            raise ValueError("each view segment must reference valid RAW indices")
        if metadata.engine == "gigaam" and len(indices) != 1:
            raise ValueError("GigaAM view segments must reference one RAW chunk")

    payload: dict[str, object] = {
        "input": str(metadata.input_path),
        "started_at": metadata.started_at.isoformat(),
        "engine": metadata.engine,
        "language": result.language,
        "requested_language": metadata.requested_language,
        "whisper_model": metadata.whisper_model,
        "gigaam_profile": metadata.gigaam_profile,
        "gigaam_model_dir": (
            str(metadata.gigaam_model_dir)
            if metadata.gigaam_model_dir is not None
            else None
        ),
        "device": metadata.device,
        "requested_device": metadata.requested_device,
        "timestamp_kind": "chunk" if metadata.engine == "gigaam" else "segment",
        "speaker_count": metadata.speaker_count,
        "initial_prompt": metadata.initial_prompt,
        "raw_segments": result.raw_segments,
        "merged_segments": [
            {"start": segment.start, "end": segment.end, "text": segment.text}
            for segment in view.segments
        ],
        "artifact_type": "transcript",
        "schema_version": 1,
        "processing": asdict(view.processing),
        "view_raw_indices": view.view_raw_indices,
    }
    if result.duration_seconds is not None:
        payload["duration_seconds"] = result.duration_seconds
    if metadata.model_identity is not None:
        payload["model_identity"] = metadata.model_identity
    return json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")


def serialize_outputs(
    result: TranscriptResult, view: TranscriptView, metadata: RunMetadata
) -> SerializedOutputs:
    timestamp_lines = ["# Transcript with Timestamps", ""]
    for segment in view.segments:
        time_range = (
            f"{format_timestamp(segment.start)}-{format_timestamp(segment.end)}"
        )
        timestamp_lines.extend([f"**{time_range}:** {segment.text}", ""])
    transcript_lines = [
        "# Transcript",
        "",
        *(segment.text for segment in view.segments),
        "",
    ]
    return SerializedOutputs(
        timestamped_markdown="\n".join(timestamp_lines).encode("utf-8"),
        markdown="\n".join(transcript_lines).encode("utf-8"),
        transcript_json=serialize_transcript(result, view, metadata),
    )


def write_outputs(out_dir: Path, outputs: SerializedOutputs) -> None:
    """Preflight all paths, then exclusively write the supplied bytes unchanged.

    This is not a transaction: a race or I/O failure can leave a partial run.
    """
    paths = output_paths(out_dir)
    for path in paths:
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"output already exists: {path}")
    contents = (
        outputs.timestamped_markdown,
        outputs.markdown,
        outputs.transcript_json,
    )
    for path, content in zip(paths, contents, strict=True):
        with path.open("xb") as output:
            output.write(content)
