"""Deterministic, conservative cleanup of diarized speaker turns.

Input groups come from diarization.speaker_segments. Only the derived presentation
turns are changed; the word-level assignments and RAW transcript stay intact.
"""

from __future__ import annotations

import re
from typing import Any

# These often are complete responses, even when surrounded by a longer turn.
_BACKCHANNELS = frozenset({
    "yeah", "right", "exactly", "no", "wow", "yes", "yep", "nope", "sure",
    "okay", "ok", "uh-huh", "mm-hmm",
    "да", "нет", "ага", "угу", "неа", "верно", "точно", "именно",
    "ого", "вау", "хорошо", "ладно", "ясно", "понятно", "окей",
})
_TERMINAL = re.compile(r"[.!?…。！？][\s\"'”’»)]*\Z")
_WORD = re.compile(r"[^\W_]+(?:[-'][^\W_]+)*", re.UNICODE)
_MAX_GAP = 0.35
_MAX_NULL_SPAN = 2.0
_MAX_NULL_WORDS = 8
_MAX_SWITCH_SPAN = 0.7
_MAX_SWITCH_CHARS = 12


def _gap(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return -0.05 <= right["start"] - left["end"] <= _MAX_GAP


def _continuation(left: dict[str, Any], right: dict[str, Any]) -> bool:
    if not _gap(left, right) or _TERMINAL.search(left["text"].rstrip()):
        return False
    first = next((char for char in right["text"] if char.isalpha()), None)
    # A lower-case continuation is useful evidence, not a guarantee of identity.
    return first is not None and first.islower()


def _standalone(text: str) -> bool:
    return text.strip().strip(".,!?…。！？\"'“”‘’«»() ").casefold() in _BACKCHANNELS


def _null_evidence(group: dict[str, Any], speaker: str) -> bool:
    if group["status"] != "unassigned":
        return False
    candidates = group.get("candidate_speakers", [])
    return not candidates or candidates == [speaker]


def _join(left: str, right: str) -> str:
    # Whisper word text normally contains its own leading space. Avoid injecting
    # spaces before punctuation, while supporting callers with stripped chunks.
    if not left or not right or left[-1].isspace() or right[0].isspace():
        return left + right
    if right[0] in ",.!?;:。！？)]}”’" or left[-1] in "([{“‘":
        return left + right
    return left + " " + right


def postprocess_speakers(groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Repair bounded A-null(s)-A and A-B-A bridges, then merge continuous turns.

    Groups without a matching speaker on *both* sides are never attributed.
    Ambiguous RTTM overlaps are not overridden. Inputs are not mutated.
    """
    turns: list[dict[str, Any]] = [
        {key: group[key] for key in ("start", "end", "text", "speaker", "status")}
        | {"candidate_speakers": list(group.get("candidate_speakers", []))}
        for group in groups
    ]
    # Work against original labels: one correction cannot justify another.
    for index, left in enumerate(groups):
        speaker = left["speaker"]
        if speaker is None or left["status"] != "assigned":
            continue
        middle = index + 1
        while middle < len(groups) and groups[middle]["speaker"] is None:
            middle += 1
        if middle > index + 1 and middle < len(groups):
            bridge = groups[index + 1:middle]
            right = groups[middle]
            if (
                right["speaker"] == speaker
                and right["status"] == "assigned"
                and right["end"] >= right["start"]
                and bridge[-1]["end"] - bridge[0]["start"] <= _MAX_NULL_SPAN
                and sum(len(_WORD.findall(part["text"])) for part in bridge)
                <= _MAX_NULL_WORDS
                and all(_null_evidence(part, speaker) for part in bridge)
                and all(not _standalone(part["text"]) for part in bridge)
                and all(
                    _continuation(a, b)
                    for a, b in zip(
                        groups[index:middle],
                        groups[index + 1:middle + 1],
                        strict=True,
                    )
                )
            ):
                for part in turns[index + 1:middle]:
                    part["speaker"] = speaker
                    part["status"] = "assigned"
            continue
        if index + 2 >= len(groups):
            continue
        middle_group, right = groups[index + 1:index + 3]
        if (
            middle_group["speaker"] is not None
            and middle_group["speaker"] != speaker
            and middle_group["status"] == right["status"] == "assigned"
            and right["speaker"] == speaker
            and middle_group["end"] - middle_group["start"] <= _MAX_SWITCH_SPAN
            and len(middle_group["text"].strip()) <= _MAX_SWITCH_CHARS
            and len(_WORD.findall(middle_group["text"])) <= 2
            and not _standalone(middle_group["text"])
            and _continuation(left, middle_group)
            and _continuation(middle_group, right)
        ):
            turns[index + 1]["speaker"] = speaker

    merged: list[dict[str, Any]] = []
    for turn in turns:
        if (
            merged
            and turn["speaker"] is not None
            and turn["status"] == merged[-1]["status"] == "assigned"
            and turn["speaker"] == merged[-1]["speaker"]
            and _continuation(merged[-1], turn)
        ):
            previous = merged[-1]
            previous["end"] = turn["end"]
            previous["text"] = _join(previous["text"], turn["text"])
        else:
            merged.append(turn.copy())
    return merged
