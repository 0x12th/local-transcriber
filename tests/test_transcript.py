from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from copy import deepcopy
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

from local_transcriber import outputs
from local_transcriber.transcript import (
    ProcessingPolicy,
    RunMetadata,
    TranscriptResult,
    process_transcript,
    select_segments,
)


def raw_fixture() -> list[dict]:
    return [
        {"start": 0, "end": 0.5, "text": "short"},
        {
            "start": 1,
            "end": 2,
            "text": " same ",
            "id": 10,
            "avg_logprob": -0.2,
            "no_speech_prob": 0.01,
            "tokens": [1, 2],
            "temperature": 0,
            "compression_ratio": 1.2,
            "extra": {"nested": ["engine-specific"]},
        },
        {"start": 2, "end": 3, "text": "Subtitles by Example"},
        {"start": 3, "end": 4, "text": "  "},
        {"start": 4, "end": 5, "text": "same"},
        {"start": 5, "end": 6, "text": "same"},
        {"start": 10, "end": 11, "text": "last"},
    ]


def metadata_fixture() -> RunMetadata:
    return RunMetadata(
        input_path=Path("synthetic.wav"),
        started_at=datetime.fromisoformat("2026-09-25T10:00:00+03:00"),
        engine="whisper",
        requested_language=None,
        device="cpu",
        requested_device="auto",
        whisper_model="turbo",
    )


class TranscriptTest(unittest.TestCase):
    def test_shared_modules_do_not_import_cli_engines_or_argparse(self) -> None:
        script = """
import builtins
import sys
original = builtins.__import__
forbidden = {'argparse', 'torch', 'whisper', 'onnxruntime', 'numpy',
             'local_transcriber.cli', 'local_transcriber.gigaam'}
def checked(name, *args, **kwargs):
    assert not any(name == x or name.startswith(x + '.') for x in forbidden), name
    return original(name, *args, **kwargs)
builtins.__import__ = checked
import local_transcriber.transcript
import local_transcriber.outputs
assert not forbidden & sys.modules.keys()
"""
        result = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True, check=False
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_selection_keeps_raw_identity_through_all_filters(self) -> None:
        result = TranscriptResult(raw_fixture(), "ru", 12.0)
        policy = ProcessingPolicy(1.0, True, 999, "none")
        selected = select_segments(result, policy)
        self.assertEqual([s.text for s in selected], ["same", "same", "same", "last"])
        self.assertEqual([s.raw_indices for s in selected], [(1,), (4,), (5,), (6,)])
        view = process_transcript(result, policy)
        self.assertEqual(view.view_raw_indices, ((1,), (4,), (5,), (6,)))
        self.assertEqual(view.segments[0].avg_logprob, -0.2)
        self.assertEqual(result.raw_segments[1]["text"], " same ")
        self.assertEqual(len(result.raw_segments), 7)

    def test_whisper_merge_carries_multiple_indices_not_equal_text_matches(
        self,
    ) -> None:
        result = TranscriptResult(raw_fixture(), "ru")
        view = process_transcript(result, ProcessingPolicy(1.0, True, 2.0, "adjacent"))
        self.assertEqual(view.view_raw_indices, ((1, 4, 5), (6,)))
        self.assertEqual([s.text for s in view.segments], ["same same same", "last"])
        self.assertEqual((view.segments[0].start, view.segments[0].end), (1, 6))
        self.assertIsNone(view.segments[0].avg_logprob)
        self.assertEqual(result.raw_segments[1]["avg_logprob"], -0.2)

    def test_empty_and_all_filtered_results_keep_empty_mapping(self) -> None:
        for raw in ([], raw_fixture()):
            for merge_policy in ("none", "adjacent"):
                with self.subTest(raw=bool(raw), merge_policy=merge_policy):
                    result = TranscriptResult(raw, "ru", 20.0)
                    policy = ProcessingPolicy(100, True, 1.5, merge_policy)
                    view = process_transcript(result, policy)
                    self.assertEqual(view.segments, ())
                    self.assertEqual(view.view_raw_indices, ())
                    self.assertEqual(result.duration_seconds, 20.0)

    def test_processing_and_serialization_do_not_mutate_inputs(self) -> None:
        engine_result = {"segments": raw_fixture(), "language": "ru"}
        before = deepcopy(engine_result)
        result = TranscriptResult.from_engine_result(engine_result)
        metadata = metadata_fixture()
        policy = ProcessingPolicy(1.0, True, 2.0)
        first = process_transcript(result, policy)
        first_bytes = outputs.serialize_transcript(result, first, metadata)
        second = process_transcript(result, policy)
        self.assertEqual(first, second)
        self.assertEqual(first.view_raw_indices, second.view_raw_indices)
        self.assertEqual(
            first_bytes, outputs.serialize_transcript(result, second, metadata)
        )
        self.assertEqual(engine_result, before)
        self.assertEqual(result.raw_segments, before["segments"])
        self.assertEqual(metadata, metadata_fixture())
        engine_result["segments"][1]["extra"]["nested"].append("caller edit")
        self.assertEqual(result.raw_segments, before["segments"])

    def test_unknown_merge_policy_is_not_silently_treated_as_none(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown merge policy"):
            policy_data = json.loads('{"merge_policy": "unknown"}')
            ProcessingPolicy(**policy_data)


class SerializationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.result = TranscriptResult(raw_fixture(), "ru")
        self.policy = ProcessingPolicy(1.0, True, 2.0)
        self.view = process_transcript(self.result, self.policy)
        self.metadata = metadata_fixture()

    def test_schema_v1_preserves_all_raw_fields_and_old_metadata(self) -> None:
        data = json.loads(
            outputs.serialize_transcript(self.result, self.view, self.metadata)
        )
        self.assertEqual(data["raw_segments"], raw_fixture())
        self.assertEqual(data["artifact_type"], "transcript")
        self.assertEqual(data["schema_version"], 1)
        self.assertEqual(data["view_raw_indices"], [[1, 4, 5], [6]])
        self.assertEqual(
            data["processing"],
            {
                "min_segment_seconds": 1.0,
                "drop_subtitle_artifacts": True,
                "merge_gap_seconds": 2.0,
                "merge_policy": "adjacent",
            },
        )
        for segment in data["merged_segments"]:
            self.assertEqual(set(segment), {"start", "end", "text"})
        self.assertEqual(data["requested_device"], "auto")
        self.assertEqual(data["device"], "cpu")
        self.assertEqual(data["whisper_model"], "turbo")
        self.assertEqual(data["started_at"], "2026-09-25T10:00:00+03:00")
        self.assertIsNone(data["requested_language"])
        self.assertIsNone(data["speaker_count"])
        self.assertIsNone(data["initial_prompt"])
        self.assertNotIn("duration_seconds", data)
        self.assertNotIn("model_identity", data)
        self.assertNotIn("source_raw_sha256", data)
        self.assertNotIn("glossary", data)

    def test_explicit_model_identity_is_copied_not_inferred_or_mutated(self) -> None:
        identity = {"verification": "unverified", "files": {"synthetic": "digest"}}
        before = deepcopy(identity)
        metadata = replace(self.metadata, model_identity=identity)
        data = json.loads(
            outputs.serialize_transcript(self.result, self.view, metadata)
        )
        self.assertEqual(identity, before)
        self.assertEqual(data["model_identity"], before)
        identity["files"]["synthetic"] = "caller edit"
        self.assertEqual(metadata.model_identity, before)

    def test_gigaam_requires_duration_and_no_merge(self) -> None:
        metadata = replace(self.metadata, engine="gigaam", whisper_model=None)
        policy = replace(self.policy, merge_policy="none")
        view = process_transcript(self.result, policy)
        with self.assertRaisesRegex(ValueError, "duration_seconds"):
            outputs.serialize_transcript(self.result, view, metadata)
        for duration in (-1, float("nan"), float("inf")):
            with (
                self.subTest(duration=duration),
                self.assertRaisesRegex(ValueError, "finite and non-negative"),
            ):
                outputs.serialize_transcript(
                    replace(self.result, duration_seconds=duration), view, metadata
                )
        result = replace(self.result, duration_seconds=12.5)
        data = json.loads(outputs.serialize_transcript(result, view, metadata))
        self.assertEqual(data["duration_seconds"], 12.5)
        self.assertEqual(data["view_raw_indices"], [[1], [4], [5], [6]])
        self.assertEqual(data["timestamp_kind"], "chunk")
        self.assertNotIn("model_identity", data)
        with self.assertRaisesRegex(ValueError, "merge_policy=none"):
            outputs.serialize_transcript(result, self.view, metadata)

    def test_empty_gigaam_keeps_full_duration_and_empty_markdown(self) -> None:
        for duration in (0.0, 31.25):
            with self.subTest(duration=duration):
                result = TranscriptResult([], "ru", duration)
                view = process_transcript(result, ProcessingPolicy(merge_policy="none"))
                metadata = replace(self.metadata, engine="gigaam", whisper_model=None)
                serialized = outputs.serialize_outputs(result, view, metadata)
                data = json.loads(serialized.transcript_json)
                self.assertEqual(data["duration_seconds"], duration)
                self.assertEqual(data["raw_segments"], [])
                self.assertEqual(data["merged_segments"], [])
                self.assertEqual(data["view_raw_indices"], [])
                self.assertEqual(serialized.markdown, b"# Transcript\n\n")
                self.assertEqual(
                    serialized.timestamped_markdown, b"# Transcript with Timestamps\n"
                )

    def test_serializes_once_and_writer_publishes_exact_returned_bytes(self) -> None:
        with patch.object(
            outputs, "serialize_transcript", wraps=outputs.serialize_transcript
        ) as spy:
            serialized = outputs.serialize_outputs(
                self.result, self.view, self.metadata
            )
        spy.assert_called_once_with(self.result, self.view, self.metadata)
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(
                outputs.json, "dumps", side_effect=AssertionError("second dump")
            ),
        ):
            root = Path(directory)
            outputs.write_outputs(root, serialized)
            self.assertEqual(
                (root / "transcript.json").read_bytes(), serialized.transcript_json
            )
            self.assertEqual((root / "transcript.md").read_bytes(), serialized.markdown)
            self.assertEqual(
                (root / "transcript_timestamps.md").read_bytes(),
                serialized.timestamped_markdown,
            )

    def test_writer_does_not_reserialize_even_nonstandard_json_bytes(self) -> None:
        exact = b'{ "synthetic": true }\n'
        with patch.object(outputs, "serialize_transcript", return_value=exact) as spy:
            serialized = outputs.serialize_outputs(
                self.result, self.view, self.metadata
            )
        spy.assert_called_once()
        self.assertIs(serialized.transcript_json, exact)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            outputs.write_outputs(root, serialized)
            self.assertEqual((root / "transcript.json").read_bytes(), exact)


class SharedWriterSafetyTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.outputs = outputs.SerializedOutputs(b"timestamped", b"plain", b"{}")

    def test_dangling_symlink_preflight_blocks_all_writes(self) -> None:
        link = self.root / "transcript.json"
        link.symlink_to(self.root / "absent")
        with self.assertRaisesRegex(FileExistsError, "already exists"):
            outputs.write_outputs(self.root, self.outputs)
        self.assertTrue(link.is_symlink())
        self.assertEqual(list(self.root.iterdir()), [link])

    def test_exclusive_create_protects_a_file_created_after_preflight(self) -> None:
        target = self.root / "transcript.json"
        real_open = Path.open

        def racing_open(path, mode="r", *args, **kwargs):
            if path == target and mode == "xb":
                with real_open(target, "wb") as other:
                    other.write(b"concurrent writer")
            return real_open(path, mode, *args, **kwargs)

        with (
            patch.object(Path, "open", racing_open),
            self.assertRaises(FileExistsError),
        ):
            outputs.write_outputs(self.root, self.outputs)
        self.assertEqual(target.read_bytes(), b"concurrent writer")

    def test_io_failure_propagates_and_does_not_claim_atomicity(self) -> None:
        target = self.root / "transcript.md"
        real_open = Path.open

        def failing_open(path, mode="r", *args, **kwargs):
            if path == target and mode == "xb":
                raise OSError("synthetic disk failure")
            return real_open(path, mode, *args, **kwargs)

        with (
            patch.object(Path, "open", failing_open),
            self.assertRaisesRegex(OSError, "synthetic disk failure"),
        ):
            outputs.write_outputs(self.root, self.outputs)
        self.assertEqual(
            (self.root / "transcript_timestamps.md").read_bytes(), b"timestamped"
        )
        self.assertFalse(target.exists())
        self.assertFalse((self.root / "transcript.json").exists())


if __name__ == "__main__":
    unittest.main()
