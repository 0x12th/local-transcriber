from __future__ import annotations

import json
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import torch

from local_transcriber.cli import (
    TranscriptSegment,
    build_initial_prompt,
    build_segments,
    choose_device,
    clean_segments,
    create_run_directory,
    format_timestamp,
    merge_adjacent_segments,
    parse_args,
    write_outputs,
)


class ProcessingTest(unittest.TestCase):
    def test_build_segments_preserves_metrics_and_normalizes_text(self) -> None:
        self.assertEqual(
            build_segments(
                {
                    "segments": [
                        {
                            "start": 0,
                            "end": 1,
                            "text": " Привет ",
                            "avg_logprob": -0.2,
                            "no_speech_prob": 0.01,
                        },
                        {"start": 1, "end": 2, "text": " "},
                        {"start": 2, "end": 3, "text": "мир"},
                    ]
                }
            ),
            [
                TranscriptSegment(0, 1, "Привет", -0.2, 0.01),
                TranscriptSegment(2, 3, "мир"),
            ],
        )

    def test_default_cleanup_preserves_speech_about_subtitles(self) -> None:
        args = parse_args(["audio.wav"])
        segments = [
            TranscriptSegment(0, 2, text)
            for text in (
                "мы добавили subtitles в продукт",
                "this service was created by our team",
                "субтитры помогают зрителям",
            )
        ]
        self.assertFalse(args.drop_subtitle_artifacts)
        self.assertEqual(
            clean_segments(segments, 0, args.drop_subtitle_artifacts), segments
        )

    def test_cleanup_filters_only_when_requested_and_keeps_duration_boundary(self):
        segments = [
            TranscriptSegment(0, 0.5, "короткий"),
            TranscriptSegment(1, 2, "речь"),
            TranscriptSegment(2, 4, "Subtitles by Example"),
        ]
        self.assertEqual(clean_segments(segments, 1, True), [segments[1]])
        self.assertEqual(clean_segments(segments, 0, False), segments)

    def test_merge_gap_boundary_and_input_immutability(self) -> None:
        segments = [
            TranscriptSegment(0, 1, "one", -0.2, 0.01),
            TranscriptSegment(2.5, 3, "two", -0.4, 0.02),
            TranscriptSegment(5, 6, "three"),
        ]
        self.assertEqual(
            merge_adjacent_segments(segments, 1.5),
            [TranscriptSegment(0, 3, "one two"), segments[2]],
        )
        self.assertEqual(segments[0].avg_logprob, -0.2)
        self.assertEqual(merge_adjacent_segments([], 1.5), [])
        self.assertEqual(merge_adjacent_segments(segments, 0), segments)

    def test_timestamp_rounding_and_hour_rollover(self) -> None:
        for seconds, expected in [
            (0, "00:00:00"),
            (59.6, "00:01:00"),
            (3599.6, "01:00:00"),
            (3661, "01:01:01"),
        ]:
            with self.subTest(seconds=seconds):
                self.assertEqual(format_timestamp(seconds), expected)

    def test_prompt_requires_opt_in_and_speaker_count(self) -> None:
        for prompt, count, enabled, expected in [
            (None, None, False, None),
            ("  ", 2, False, None),
            (" Terms ", 2, False, "Terms"),
            (" Terms ", None, True, "Terms"),
            (" Terms ", 2, True, "Terms The recording contains 2 speakers."),
            (None, 1, True, "The recording contains 1 speakers."),
        ]:
            with self.subTest(prompt=prompt, count=count, enabled=enabled):
                self.assertEqual(build_initial_prompt(prompt, count, enabled), expected)

    def test_device_selection_matches_available_hardware(self) -> None:
        cuda = torch.cuda.is_available()
        mps = torch.backends.mps.is_available()
        self.assertEqual(choose_device("cpu"), "cpu")
        self.assertEqual(
            choose_device("auto"), "cuda" if cuda else "mps" if mps else "cpu"
        )
        for device, available in [("cuda", cuda), ("mps", mps)]:
            with self.subTest(device=device):
                if available:
                    self.assertEqual(choose_device(device), device)
                else:
                    with self.assertRaisesRegex(ValueError, "not available"):
                        choose_device(device)


class OutputTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.out = Path(self.temp.name)
        self.input = self.out / "meeting.wav"
        self.started_at = datetime.fromisoformat("2026-09-12T14:30:05+03:00")
        self.args = parse_args([str(self.input)])
        self.result = {
            "language": "ru",
            "segments": [
                {
                    "start": 0,
                    "end": 1,
                    "text": " Привет ",
                    "avg_logprob": -0.2,
                    "no_speech_prob": 0.01,
                },
                {
                    "start": 1,
                    "end": 2,
                    "text": "мир",
                    "avg_logprob": -0.4,
                    "no_speech_prob": 0.02,
                },
                {"start": 5, "end": 6, "text": "Subtitles by Example"},
            ],
        }
        self.segments = merge_adjacent_segments(
            clean_segments(build_segments(self.result), 0, True), 1.5
        )

    def write(self) -> Path:
        run_dir = create_run_directory(self.out, self.input, self.started_at)
        write_outputs(
            run_dir,
            self.input,
            self.segments,
            self.result,
            self.args,
            None,
            "cpu",
            self.started_at,
        )
        return run_dir

    def test_outputs_keep_raw_metrics_and_only_text_ranges_for_merged(self) -> None:
        run_dir = self.write()
        self.assertEqual(run_dir, self.out / "meeting" / "2026-09-12_14-30-05")
        data = json.loads((run_dir / "transcript.json").read_text())
        self.assertEqual(data["started_at"], "2026-09-12T14:30:05+03:00")
        self.assertEqual(data["raw_segments"], self.result["segments"])
        self.assertEqual(
            data["merged_segments"], [{"start": 0, "end": 2, "text": "Привет мир"}]
        )
        self.assertNotIn("segments", data)
        self.assertEqual(data["language"], "ru")
        self.assertEqual(
            (run_dir / "transcript.md").read_text(), "# Transcript\n\nПривет мир\n"
        )
        self.assertIn(
            "**00:00:00-00:00:02:** Привет мир",
            (run_dir / "transcript_timestamps.md").read_text(),
        )

    def test_any_existing_output_blocks_all_writes(self) -> None:
        for name in ("transcript.md", "transcript_timestamps.md", "transcript.json"):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                self.out = Path(directory)
                existing = self.out / name
                existing.write_text("keep me")
                with self.assertRaisesRegex(FileExistsError, "already exists"):
                    write_outputs(
                        self.out,
                        self.input,
                        self.segments,
                        self.result,
                        self.args,
                        None,
                        "cpu",
                        self.started_at,
                    )
                self.assertEqual(existing.read_text(), "keep me")
                self.assertEqual(list(self.out.iterdir()), [existing])

    def test_repeated_runs_preserve_previous_results(self) -> None:
        first = self.write()
        (first / "transcript.md").write_text("old")
        second = self.write()
        third = self.write()
        self.assertEqual(second.name, "2026-09-12_14-30-05-2")
        self.assertEqual(third.name, "2026-09-12_14-30-05-3")
        self.assertEqual((first / "transcript.md").read_text(), "old")
        self.assertIn("Привет мир", (second / "transcript.md").read_text())

    def test_input_named_like_output_is_preserved(self) -> None:
        self.input = self.out / "transcript.json"
        self.input.write_text("source")
        self.write()
        self.assertEqual(self.input.read_text(), "source")

    def test_distinct_input_stems_coexist(self) -> None:
        first = self.write()
        self.input = self.out / "another.wav"
        second = self.write()
        self.assertEqual(first.parent, self.out / "meeting")
        self.assertEqual(second.parent, self.out / "another")
        self.assertTrue((first / "transcript.json").is_file())
        self.assertTrue((second / "transcript.json").is_file())

    def test_same_stem_from_different_sources_gets_separate_runs(self) -> None:
        first = self.write()
        self.input = self.out / "other" / "meeting.mp4"
        second = self.write()
        self.assertEqual(first.parent, second.parent)
        self.assertNotEqual(first, second)

    def test_concurrent_runs_reserve_unique_directories(self) -> None:
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(self.write) for _ in range(8)]
            paths = [future.result() for future in futures]
        self.assertEqual(len(set(paths)), 8)
        self.assertEqual(
            {path.name for path in paths},
            {"2026-09-12_14-30-05"}
            | {f"2026-09-12_14-30-05-{suffix}" for suffix in range(2, 9)},
        )
        for path in paths:
            self.assertIn("Привет мир", (path / "transcript.md").read_text())
