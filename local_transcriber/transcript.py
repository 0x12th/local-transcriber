"""Engine-independent RAW results and deterministic transcript views.

RAW payloads retain engine-specific fields. Processing never edits these payloads;
source indices travel with the view rather than being recovered from text.
"""

from __future__ import annotations

import re
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

SUBTITLE_ARTIFACT_RE = re.compile(
    r"(редактор субтитров|корректор|субтитры|subtitles?|caption|captions?|"
    r"created by|transcribed by)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class TranscriptResult:
    """Owned snapshot of the RAW segments; consumers must treat it as read-only."""

    raw_segments: list[dict[str, Any]]
    language: str | None
    duration_seconds: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "raw_segments", deepcopy(self.raw_segments))

    @classmethod
    def from_engine_result(cls, result: dict[str, Any]) -> TranscriptResult:
        return cls(
            raw_segments=result["segments"],
            language=result.get("language"),
            duration_seconds=result.get("duration_seconds"),
        )


@dataclass(frozen=True)
class RunMetadata:
    input_path: Path
    started_at: datetime
    engine: Literal["whisper", "gigaam"]
    requested_language: str | None
    device: str
    requested_device: str
    whisper_model: str | None = None
    gigaam_profile: str | None = None
    gigaam_model_dir: Path | None = None
    speaker_count: int | None = None
    initial_prompt: str | None = None
    # Populated by a verified model loader, never inferred from a directory name.
    model_identity: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "model_identity", deepcopy(self.model_identity))


@dataclass(frozen=True)
class ProcessingPolicy:
    min_segment_seconds: float = 0.0
    drop_subtitle_artifacts: bool = False
    merge_gap_seconds: float = 1.5
    merge_policy: Literal["none", "adjacent"] = "adjacent"

    def __post_init__(self) -> None:
        if self.merge_policy not in ("none", "adjacent"):
            raise ValueError(f"unknown merge policy: {self.merge_policy}")


@dataclass(frozen=True)
class TranscriptSegment:
    start: float
    end: float
    text: str
    avg_logprob: float | None = None
    no_speech_prob: float | None = None
    # Keep legacy value equality while carrying provenance through every transform.
    raw_indices: tuple[int, ...] = field(default=(), compare=False)


@dataclass(frozen=True)
class TranscriptView:
    segments: tuple[TranscriptSegment, ...]
    processing: ProcessingPolicy

    @property
    def view_raw_indices(self) -> tuple[tuple[int, ...], ...]:
        return tuple(segment.raw_indices for segment in self.segments)


def maybe_float(value: Any) -> float | None:
    return None if value is None else float(value)


def build_segments(result: dict[str, Any]) -> list[TranscriptSegment]:
    """Compatibility entry point for engine dictionaries, preserving RAW indices."""
    segments = []
    for index, segment in enumerate(result["segments"]):
        text = str(segment["text"]).strip()
        if text:
            segments.append(
                TranscriptSegment(
                    start=float(segment["start"]),
                    end=float(segment["end"]),
                    text=text,
                    avg_logprob=maybe_float(segment.get("avg_logprob")),
                    no_speech_prob=maybe_float(segment.get("no_speech_prob")),
                    raw_indices=(index,),
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
                raw_indices=previous.raw_indices + segment.raw_indices,
            )
        else:
            merged.append(segment)
    return merged


def select_segments(
    result: TranscriptResult, policy: ProcessingPolicy
) -> list[TranscriptSegment]:
    """Select unmerged segments; raw_indices address the exact, unstripped RAW text."""
    return clean_segments(
        build_segments({"segments": result.raw_segments}),
        policy.min_segment_seconds,
        policy.drop_subtitle_artifacts,
    )


def process_transcript(
    result: TranscriptResult, policy: ProcessingPolicy
) -> TranscriptView:
    segments = select_segments(result, policy)
    if policy.merge_policy == "adjacent":
        segments = merge_adjacent_segments(segments, policy.merge_gap_seconds)
    return TranscriptView(tuple(segments), policy)
