"""Offline checks for conservative, provenance-preserving speaker annotations."""

from __future__ import annotations

import subprocess
import sys
import unittest
from copy import deepcopy
from typing import Any, cast

from local_transcriber.diarization import (
    parse_rttm,
    speaker_items,
    speaker_markdown,
    speaker_segments,
)


def rttm(start: str, duration: str, speaker: str = "speaker_1") -> str:
    return f"SPEAKER recording 1 {start} {duration} <NA> <NA> {speaker} <NA> <NA>"


class ParseRttmTest(unittest.TestCase):
    def test_preserves_overlaps_input_order_and_exact_raw_lines(self) -> None:
        first = rttm("0.0", "1.5")
        second = "  " + rttm("1.0", "1.0", "speaker_2") + "  "
        self.assertEqual(
            parse_rttm(first + "\n" + second, "recording", 2.0),
            [
                {
                    "index": 0,
                    "start": 0.0,
                    "end": 1.5,
                    "duration": 1.5,
                    "speaker": "speaker_1",
                    "raw_line": first,
                },
                {
                    "index": 1,
                    "start": 1.0,
                    "end": 2.0,
                    "duration": 1.0,
                    "speaker": "speaker_2",
                    "raw_line": second,
                },
            ],
        )
        self.assertEqual(parse_rttm("", "recording", 2.0), [])
        self.assertEqual(
            parse_rttm(rttm("1.0", "1.05"), "recording", 2.0)[0]["end"], 2.05
        )

    def test_rejects_invalid_format_times_and_bounds(self) -> None:
        bad_lines = [
            rttm("0", "1").replace("SPEAKER", "LEXEME"),
            rttm("0", "1").replace("recording", "other"),
            rttm("0", "1").replace("recording 1", "recording 2"),
            rttm("0", "1").rsplit(" ", 1)[0],
            rttm("NaN", "1"),
            rttm("inf", "1"),
            rttm("-0.1", "1"),
            rttm("0", "NaN"),
            rttm("0", "inf"),
            rttm("0", "0"),
            rttm("0", "-1"),
            rttm("1", "1.051"),
            rttm("1e308", "1e308"),
            rttm("0", "1", "invalid/id"),
            rttm("0", "1", "speaker_X"),
            "",
        ]
        for line in bad_lines:
            with (
                self.subTest(line=line),
                self.assertRaisesRegex(ValueError, "RTTM .*line"),
            ):
                parse_rttm(
                    rttm("0", "1") + "\n" + line + "\n" + rttm("1", "1"),
                    "recording",
                    2.0,
                )
        for duration in (-1, float("nan"), float("inf")):
            with (
                self.subTest(duration=duration),
                self.assertRaisesRegex(ValueError, "recording duration"),
            ):
                parse_rttm("", "recording", duration)


class SpeakerMappingTest(unittest.TestCase):
    def test_sequential_speakers_inside_one_raw_segment(self) -> None:
        raw = [
            {
                "start": 0.0,
                "end": 2.0,
                "text": "Canonical ASR text",
                "words": [
                    {"start": 0, "end": 0.5, "word": " Hi"},
                    {"start": 0.5, "end": 1, "word": ", friend"},
                    {"start": 1, "end": 1.5, "word": " bye"},
                    {"start": 1.5, "end": 2, "word": "!"},
                ],
            }
        ]
        original = deepcopy(raw)
        intervals = parse_rttm(
            rttm("0", "1") + "\n" + rttm("1", "1", "speaker_2"), "recording", 2
        )
        items = speaker_items(raw, intervals)
        self.assertEqual(raw, original)
        self.assertEqual(
            [item["text"] for item in items], [" Hi", ", friend", " bye", "!"]
        )
        self.assertEqual([item["raw_word_index"] for item in items], [0, 1, 2, 3])
        self.assertEqual([item["raw_segment_index"] for item in items], [0] * 4)
        self.assertEqual([item["status"] for item in items], ["assigned"] * 4)
        self.assertEqual(
            [item["speaker"] for item in items], ["speaker_1"] * 2 + ["speaker_2"] * 2
        )
        self.assertEqual(
            [item["candidate_speakers"] for item in items],
            [["speaker_1"]] * 2 + [["speaker_2"]] * 2,
        )
        self.assertEqual(
            [item["rttm_interval_indices"] for item in items], [[0], [0], [1], [1]]
        )
        self.assertEqual([item["reason"] for item in items], [None] * 4)
        groups = speaker_segments(items)
        self.assertEqual([group["text"] for group in groups], [" Hi, friend", " bye!"])
        self.assertEqual(
            [group["raw_word_indices"] for group in groups], [[0, 1], [2, 3]]
        )
        self.assertEqual([group["item_indices"] for group in groups], [[0, 1], [2, 3]])
        self.assertEqual(
            [group["speaker"] for group in groups], ["speaker_1", "speaker_2"]
        )
        self.assertEqual([group["start"] for group in groups], [0, 1])
        self.assertEqual([group["end"] for group in groups], [1, 2])
        self.assertEqual(raw[0]["text"], "Canonical ASR text")

    def test_overlap_partial_gaps_and_same_speaker_union(self) -> None:
        raw: list[dict[str, Any]] = [
            {
                "text": "original",
                "words": [
                    {"start": 0, "end": 1, "word": " overlap"},
                    {"start": 1, "end": 2, "word": " gap"},
                    {"start": 2, "end": 3, "word": " joined"},
                    {"start": 3, "end": 4, "word": " outside"},
                    {"start": 4, "end": 4, "word": " zero"},
                ],
            }
        ]
        intervals = parse_rttm(
            "\n".join(
                [
                    rttm("0", "0.8"),
                    rttm("0.8", "0.1", "speaker_2"),
                    rttm("1", "0.4"),
                    rttm("1.6", "0.4"),
                    rttm("2", "0.5"),
                    rttm("2.5", "0.5"),
                ]
            ),
            "recording",
            4,
        )
        items = speaker_items(raw, intervals)
        self.assertEqual(
            [item["status"] for item in items],
            ["ambiguous", "unassigned", "assigned", "unassigned", "unassigned"],
        )
        self.assertEqual(
            [item["reason"] for item in items],
            [
                "multiple_speakers_overlap_word",
                "partial_rttm_coverage",
                None,
                "no_rttm_coverage",
                "zero_duration",
            ],
        )
        self.assertEqual(
            [item["speaker"] for item in items], [None, None, "speaker_1", None, None]
        )
        self.assertEqual(
            [item["candidate_speakers"] for item in items],
            [["speaker_1", "speaker_2"], ["speaker_1"], ["speaker_1"], [], []],
        )
        self.assertEqual(
            [item["rttm_interval_indices"] for item in items],
            [[0, 1], [2, 3], [4, 5], [], []],
        )
        groups = speaker_segments(items)
        self.assertEqual(
            [group["text"] for group in groups],
            [" overlap", " gap", " joined", " outside zero"],
        )
        self.assertEqual(groups[-1]["raw_word_indices"], [3, 4])
        self.assertEqual(groups[-1]["reasons"], ["no_rttm_coverage", "zero_duration"])
        self.assertEqual(
            "".join(group["text"] for group in groups),
            "".join(
                            word["word"]
                            for word in cast(list[dict[str, str]], raw[0]["words"])
                        ),
        )

    def test_empty_intervals_fallback_and_raw_boundaries(self) -> None:
        raw = [
            {"start": 0, "end": 1, "text": " Untimed original"},
            {"start": 1, "end": 2, "text": " separate", "words": []},
            {
                "start": 2,
                "end": 3,
                "text": " canonical",
                "words": [{"start": 2, "end": 3, "word": " word"}],
            },
        ]
        items = speaker_items(raw, [])
        self.assertEqual(
            [item["text"] for item in items],
            [" Untimed original", " separate", " word"],
        )
        self.assertEqual([item["raw_word_index"] for item in items], [None, None, 0])
        self.assertEqual(
            [item["reason"] for item in items],
            ["no_word_timestamps", "no_word_timestamps", "no_rttm_coverage"],
        )
        self.assertEqual(
            [item["rttm_interval_indices"] for item in items], [[], [], []]
        )
        groups = speaker_segments(items)
        self.assertEqual([group["raw_segment_index"] for group in groups], [0, 1, 2])
        self.assertEqual(
            [group["raw_word_indices"] for group in groups], [[None], [None], [0]]
        )
        self.assertEqual(
            [group["text"] for group in groups],
            [" Untimed original", " separate", " word"],
        )
        self.assertEqual(speaker_segments([]), [])
        self.assertEqual(speaker_items([], []), [])

    def test_invalid_word_timestamps_or_text_are_rejected(self) -> None:
        for word in (
            {"start": 0, "end": 1},
            {"start": float("nan"), "end": 1, "word": "bad"},
            {"start": 1, "end": 0, "word": "bad"},
        ):
            with self.subTest(word=word), self.assertRaises(ValueError):
                speaker_items([{"words": [word]}], [])


class SpeakerMarkdownTest(unittest.TestCase):
    def test_readable_labels_times_utf8_and_failure(self) -> None:
        segments = speaker_segments(
            speaker_items(
                [
                    {
                        "start": 0,
                        "end": 1,
                        "text": " Здравствуйте",
                        "words": [{"start": 0, "end": 1, "word": " Здравствуйте"}],
                    },
                    {
                        "start": 1,
                        "end": 2,
                        "text": " maybe",
                        "words": [{"start": 1, "end": 2, "word": " maybe"}],
                    },
                    {"start": 2, "end": 3, "text": " unknown"},
                ],
                parse_rttm(
                    rttm("0", "2") + "\n" + rttm("1.5", "0.5", "speaker_2"),
                    "recording",
                    3,
                ),
            )
        )
        markdown = speaker_markdown(segments, "success").decode("utf-8")
        self.assertIn("00:00:00.00–00:00:01.00 speaker_1:**  Здравствуйте", markdown)
        self.assertIn(
            "00:00:01.00–00:00:02.00 ? (ambiguous: speaker_1, speaker_2):**  maybe",
            markdown,
        )
        self.assertIn("00:00:02.00–00:00:03.00 ? (unassigned):**  unknown", markdown)
        self.assertNotIn("FAILED", markdown)
        self.assertIn(
            "DIARIZATION FAILED", speaker_markdown(segments, "failed").decode("utf-8")
        )
        self.assertIn(
            "DIARIZATION FAILED", speaker_markdown([], "failed").decode("utf-8")
        )
        with self.assertRaisesRegex(ValueError, "unknown diarization status"):
            speaker_markdown([], "skipped")

    def test_module_import_has_no_model_dependencies(self) -> None:
        script = """
import builtins
import sys
original = builtins.__import__
forbidden = {'torch', 'whisper', 'numpy', 'local_transcriber.cli'}
def checked(name, *args, **kwargs):
    assert not any(name == x or name.startswith(x + '.') for x in forbidden), name
    return original(name, *args, **kwargs)
builtins.__import__ = checked
import local_transcriber.diarization
assert not forbidden & sys.modules.keys()
"""
        result = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True, check=False
        )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
