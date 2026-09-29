"""Offline checks for the shared speaker processing helpers."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

from local_transcriber import speakers_cli


class SpeakerProcessingTest(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.wav = Path(temporary.name) / "audio.wav"
        self.model = Path(temporary.name) / "model.gguf"
        self.executable = Path(temporary.name) / "nemo-speech"
        self.segments: list[dict[str, Any]] = [
            {"start": 0.0, "end": 2.0, "text": " Hello world!", "words": [
                {"start": 0.0, "end": 1.0, "word": " Hello"},
                {"start": 1.0, "end": 2.0, "word": " world!"},
            ]},
        ]
        self.intervals = [
            {"index": 0, "start": 0.0, "end": 1.0, "speaker": "speaker_7"},
            {"index": 1, "start": 1.0, "end": 2.0, "speaker": "speaker_3"},
        ]

    def test_groups_words_into_compact_turns_with_first_seen_speaker_ids(self) -> None:
        with patch.object(
            speakers_cli, "_diarize", return_value=("", self.intervals)
        ) as diarize:
            markdown, payload, error = speakers_cli.diarize_transcript(
                self.wav, 2.0, self.segments, self.model, self.executable
            )
        diarize.assert_called_once_with(
            self.executable, self.model, self.wav, 2.0, "metal", 3600
        )
        self.assertIsNone(error)
        self.assertIn(b"speaker_7", markdown)
        self.assertEqual(json.loads(payload), [
            {"start": 0.0, "end": 1.0, "text": "Hello", "speaker": 1},
            {"start": 1.0, "end": 2.0, "text": "world!", "speaker": 2},
        ])
        self.assertTrue(all(set(turn) == {"start", "end", "text", "speaker"}
                            for turn in json.loads(payload)))

    def test_overlapping_speakers_and_uncovered_words_are_unassigned(self) -> None:
        intervals = [
            {"index": 0, "start": 0.0, "end": 1.0, "speaker": "speaker_3"},
            {"index": 1, "start": 0.5, "end": 1.5, "speaker": "speaker_7"},
        ]
        self.segments[0]["words"] = [
            {"start": 0.0, "end": 0.5, "word": " First"},
            {"start": 0.5, "end": 1.0, "word": " maybe"},
            {"start": 1.0, "end": 1.5, "word": " next"},
            {"start": 1.5, "end": 2.0, "word": " unknown"},
        ]
        with patch.object(speakers_cli, "_diarize", return_value=("", intervals)):
            _, payload, error = speakers_cli.diarize_transcript(
                self.wav, 2.0, self.segments, self.model, self.executable
            )
        self.assertIsNone(error)
        self.assertEqual([turn["speaker"] for turn in json.loads(payload)],
                         [1, None, 2, None])

    def test_failed_diarizer_retains_unlabeled_words(self) -> None:
        with patch.object(
            speakers_cli, "_diarize", side_effect=RuntimeError("offline failure")
        ):
            markdown, payload, error = speakers_cli.diarize_transcript(
                self.wav, 2.0, self.segments, self.model, self.executable
            )
        self.assertEqual(error, "offline failure")
        self.assertIn(b"DIARIZATION FAILED", markdown)
        self.assertEqual(json.loads(payload), [
            {"start": 0.0, "end": 2.0, "text": "Hello world!", "speaker": None},
        ])

    def test_invalid_rttm_is_reported_without_labels(self) -> None:
        def run(command: list[str], timeout: int) -> str:
            Path(command[command.index("--output") + 1]).write_text("not RTTM\n")
            return ""

        with patch.object(speakers_cli, "_run", side_effect=run):
            markdown, payload, error = speakers_cli.diarize_transcript(
                self.wav, 2.0, self.segments, self.model, self.executable
            )
        assert error is not None
        self.assertIn("invalid RTTM line", error)
        self.assertIn(b"DIARIZATION FAILED", markdown)
        self.assertIsNone(json.loads(payload)[0]["speaker"])

    def test_word_timestamp_beyond_audio_duration_signals_failure(self) -> None:
        self.segments[0]["words"][1]["end"] = 3.0
        with patch.object(speakers_cli, "_diarize", return_value=("", self.intervals)):
            markdown, payload, error = speakers_cli.diarize_transcript(
                self.wav, 2.0, self.segments, self.model, self.executable
            )
        assert error is not None
        self.assertIn("exceeds audio duration", error)
        self.assertIn(b"DIARIZATION FAILED", markdown)
        self.assertIsNone(json.loads(payload)[0]["speaker"])

    def test_whisper_uses_named_model_word_timestamps_and_prompt(self) -> None:
        import whisper

        with patch.object(whisper, "load_model") as load:
            load.return_value.transcribe.return_value = {"segments": []}
            result = speakers_cli._transcribe(
                self.wav, "turbo", "cpu", "en", "Names and terms"
            )
        self.assertEqual(result, {"segments": []})
        load.assert_called_once_with("turbo", device="cpu")
        load.return_value.transcribe.assert_called_once_with(
            str(self.wav), task="transcribe", verbose=False,
            word_timestamps=True, fp16=False, language="en",
            initial_prompt="Names and terms",
        )

    def test_mps_alignment_moves_dtw_input_to_cpu(self) -> None:
        import torch
        import whisper
        import whisper.timing

        class Alignment:
            def cpu(self):
                return "cpu alignment"

        def transcribe(*args, **kwargs):
            self.assertEqual(
                whisper.timing.dtw(cast(torch.Tensor, Alignment())), "aligned"
            )
            return {"segments": []}

        with (
            patch.object(whisper, "load_model") as load,
            patch.object(whisper.timing, "dtw", return_value="aligned") as dtw,
            patch.object(torch.mps, "empty_cache") as empty_cache,
        ):
            load.return_value.transcribe.side_effect = transcribe
            speakers_cli._transcribe(self.wav, "turbo", "mps", None)
            dtw.assert_called_once_with("cpu alignment")
            empty_cache.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
