"""Pure, conservative speaker annotations over canonical Whisper RAW segments."""

from __future__ import annotations

import math
import re
from typing import Any

_SPEAKER_ID = re.compile(r"speaker_[0-9]+\Z")


def _finite_time(value: Any, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"invalid {label}: {value!r}") from error
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"invalid {label}: {value!r}")
    return number


def parse_rttm(
    raw: str, recording_id: str, duration_seconds: float
) -> list[dict[str, Any]]:
    """Parse channel-1 SPEAKER records, retaining input order and raw lines."""
    duration_limit = _finite_time(duration_seconds, "recording duration")
    intervals: list[dict[str, Any]] = []
    for line_number, line in enumerate(raw.splitlines(), 1):
        fields = line.split()
        if (
            len(fields) != 10
            or fields[0] != "SPEAKER"
            or fields[1] != recording_id
            or fields[2] != "1"
            or _SPEAKER_ID.fullmatch(fields[7]) is None
        ):
            raise ValueError(f"invalid RTTM line {line_number}: {line!r}")
        start = _finite_time(fields[3], f"RTTM start at line {line_number}")
        duration = _finite_time(fields[4], f"RTTM duration at line {line_number}")
        end = start + duration
        if duration == 0 or not math.isfinite(end) or end > duration_limit + 0.05:
            raise ValueError(f"invalid RTTM bounds at line {line_number}: {line!r}")
        intervals.append(
            {
                "index": len(intervals),
                "start": start,
                "end": end,
                "duration": duration,
                "speaker": fields[7],
                "raw_line": line,
            }
        )
    return intervals


def _label_span(
    start: float, end: float, intervals: list[dict[str, Any]]
) -> dict[str, Any]:
    if end <= start:
        return {
            "status": "unassigned",
            "speaker": None,
            "candidate_speakers": [],
            "rttm_interval_indices": [],
            "reason": "zero_duration",
        }
    touched = [
        interval
        for interval in intervals
        if interval["start"] < end and interval["end"] > start
    ]
    candidates = sorted({interval["speaker"] for interval in touched})
    result: dict[str, Any] = {
        "speaker": None,
        "candidate_speakers": candidates,
        "rttm_interval_indices": [interval["index"] for interval in touched],
    }
    if len(candidates) > 1:
        result.update(status="ambiguous", reason="multiple_speakers_overlap_word")
    elif not candidates:
        result.update(status="unassigned", reason="no_rttm_coverage")
    else:
        # Same-speaker intervals can jointly cover a word, but never bridge a gap.
        covered_until = start
        for interval in sorted(touched, key=lambda entry: entry["start"]):
            if interval["start"] > covered_until:
                break
            covered_until = max(covered_until, interval["end"])
        if covered_until >= end:
            result.update(status="assigned", speaker=candidates[0], reason=None)
        else:
            result.update(status="unassigned", reason="partial_rttm_coverage")
    return result


def speaker_items(
    raw_segments: list[dict[str, Any]], intervals: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Annotate each Whisper word; keep untimed segment text as an unassigned item."""
    items: list[dict[str, Any]] = []
    for segment_index, segment in enumerate(raw_segments):
        words = segment.get("words") or []
        for word_index, word in enumerate(words):
            if not isinstance(word.get("word"), str):
                raise ValueError(
                    f"missing Whisper word text at {segment_index}:{word_index}"
                )
            start = _finite_time(word.get("start"), "word start")
            end = _finite_time(word.get("end"), "word end")
            if end < start:
                raise ValueError(f"word {segment_index}:{word_index} ends before start")
            items.append(
                {
                    "start": start,
                    "end": end,
                    "text": word["word"],
                    "raw_segment_index": segment_index,
                    "raw_word_index": word_index,
                    **_label_span(start, end, intervals),
                }
            )
        if not words and segment.get("text"):
            start = _finite_time(segment.get("start"), "segment start")
            end = _finite_time(segment.get("end"), "segment end")
            if end < start:
                raise ValueError(f"segment {segment_index} ends before start")
            items.append(
                {
                    "start": start,
                    "end": end,
                    "text": segment["text"],
                    "raw_segment_index": segment_index,
                    "raw_word_index": None,
                    "status": "unassigned",
                    "speaker": None,
                    "candidate_speakers": [],
                    "rttm_interval_indices": [],
                    "reason": "no_word_timestamps",
                }
            )
    return items


def speaker_segments(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Group adjacent compatible items without crossing RAW segment boundaries."""
    segments: list[dict[str, Any]] = []
    for item_index, item in enumerate(items):
        if segments and all(
            segments[-1][key] == item[key]
            for key in ("raw_segment_index", "status", "speaker", "candidate_speakers")
        ):
            group = segments[-1]
            group["end"] = max(group["end"], item["end"])
            group["text"] += item["text"]
            group["item_indices"].append(item_index)
            group["raw_word_indices"].append(item["raw_word_index"])
            for index in item["rttm_interval_indices"]:
                if index not in group["rttm_interval_indices"]:
                    group["rttm_interval_indices"].append(index)
            if item["reason"] not in group["reasons"]:
                group["reasons"].append(item["reason"])
        else:
            segments.append(
                {
                    "start": item["start"],
                    "end": item["end"],
                    "text": item["text"],
                    "raw_segment_index": item["raw_segment_index"],
                    "raw_word_indices": [item["raw_word_index"]],
                    "item_indices": [item_index],
                    "status": item["status"],
                    "speaker": item["speaker"],
                    "candidate_speakers": list(item["candidate_speakers"]),
                    "rttm_interval_indices": list(item["rttm_interval_indices"]),
                    "reasons": [item["reason"]],
                }
            )
    return segments


def _timestamp(seconds: float) -> str:
    # Centiseconds keep short words legible while avoiding second-level rounding.
    centiseconds = round(seconds * 100)
    hours, remainder = divmod(centiseconds, 360000)
    minutes, remainder = divmod(remainder, 6000)
    seconds, fraction = divmod(remainder, 100)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{fraction:02d}"


def speaker_markdown(segments: list[dict[str, Any]], diarization_status: str) -> bytes:
    """Render derived annotations; the RAW transcript remains authoritative."""
    if diarization_status not in ("success", "failed"):
        raise ValueError(f"unknown diarization status: {diarization_status!r}")
    lines = ["# Speaker Transcript"]
    if diarization_status == "failed":
        lines.extend(["", "**DIARIZATION FAILED — speaker labels are unavailable.**"])
    for segment in segments:
        status = segment["status"]
        if status == "assigned":
            label = segment["speaker"]
        elif status == "ambiguous":
            label = f"? (ambiguous: {', '.join(segment['candidate_speakers'])})"
        elif status == "unassigned":
            label = "? (unassigned)"
        else:
            raise ValueError(f"unknown speaker status: {status!r}")
        lines.extend(
            [
                "",
                f"**{_timestamp(segment['start'])}–{_timestamp(segment['end'])} "
                f"{label}:** {segment['text']}",
            ]
        )
    return ("\n".join(lines) + "\n").encode("utf-8")
