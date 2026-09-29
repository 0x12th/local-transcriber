"""Hermetic end-to-end contract for transcribe --diarize."""

from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from local_transcriber import cli, speakers_cli
from local_transcriber.models.nemotron import NemotronInstallError


class DiarizeCliTest(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / "recording.wav"
        self.source.write_bytes(b"fixture (decoder mocked)")
        self.out_dir = self.root / "out"
        self.model = self.root / "nemotron.gguf"
        self.executable = self.root / "nemo-speech"
        self.segments = [
            {"id": 7, "start": 0.0, "end": 1.0, "text": " Hello", "words": [
                {"start": 0.0, "end": 1.0, "word": " Hello"},
            ]},
            {"id": 8, "start": 1.0, "end": 2.0, "text": " world!", "words": [
                {"start": 1.0, "end": 2.0, "word": " world!"},
            ]},
        ]
        self.intervals = [
            {"index": 0, "start": 0.0, "end": 1.0, "speaker": "speaker_1"},
            {"index": 1, "start": 1.0, "end": 2.0, "speaker": "speaker_2"},
        ]

    def args(self, *extra: str) -> list[str]:
        return [str(self.source), "--diarize", "--device", "cpu",
                "--out-dir", str(self.out_dir), *extra]

    def invoke(self, *extra: str, failure: Exception | None = None):
        stdout, stderr = io.StringIO(), io.StringIO()
        with (
            patch("local_transcriber.models.nemotron.installed_nemotron",
                  return_value=(self.model, self.executable)) as installed,
            patch.object(speakers_cli, "_decode", return_value=2.0) as decode,
            patch.object(
                speakers_cli, "_transcribe",
                return_value={"segments": self.segments, "language": "en"},
            ) as asr,
            patch.object(
                speakers_cli, "_diarize",
                return_value=("", self.intervals), side_effect=failure,
            ) as diarize,
            patch("local_transcriber.models.nemotron.install_nemotron") as installer,
            redirect_stdout(stdout), redirect_stderr(stderr),
        ):
            try:
                code = cli.main(self.args(*extra))
            except SystemExit as error:
                code = error.code
            installed.assert_called_once_with()
            installer.assert_not_called()
        return code, stdout.getvalue(), stderr.getvalue(), decode, asr, diarize

    def run_dir(self) -> Path:
        runs = list((self.out_dir / self.source.stem).iterdir())
        self.assertEqual(len(runs), 1)
        self.assertEqual({path.name for path in runs[0].iterdir()}, {
            "transcript.md", "transcript_timestamps.md", "transcript_speakers.md",
            "transcript.json",
        })
        return runs[0]

    def test_success_reuses_one_wav_and_preserves_plain_markdown(self) -> None:
        code, stdout, stderr, decode, asr, diarize = self.invoke(
            "--language", "en", "--initial-prompt", "Terms", "--speaker-count", "5",
        )
        self.assertIsNone(code)
        self.assertNotIn("failed", stderr.lower())
        run = self.run_dir()
        wav = decode.call_args.args[1]
        self.assertEqual(asr.call_args.args, (wav, "turbo", "cpu", "en", "Terms"))
        self.assertEqual(diarize.call_args.args[:3], (self.executable, self.model, wav))
        self.assertEqual(diarize.call_args.args[3:5], (2.0, "metal"))
        self.assertIn(str(run / "transcript.json"), stdout)
        self.assertEqual((run / "transcript.md").read_text(),
                         "# Transcript\n\nHello world!\n")
        self.assertEqual(
            (run / "transcript_timestamps.md").read_text(),
            "# Transcript with Timestamps\n\n"
            "**00:00:00-00:00:02:** Hello world!\n",
        )
        self.assertIn(
            "speaker_1:**  Hello", (run / "transcript_speakers.md").read_text()
        )
        self.assertEqual(json.loads((run / "transcript.json").read_text()), [
            {"start": 0.0, "end": 1.0, "text": "Hello", "speaker": 1},
            {"start": 1.0, "end": 2.0, "text": "world!", "speaker": 2},
        ])
        self.assertEqual(set(json.loads((run / "transcript.json").read_text())[0]),
                         {"start", "end", "text", "speaker"})

    def test_processing_flags_apply_to_plain_markdown(self) -> None:
        code, _, _, _, _, _ = self.invoke("--min-segment-seconds", "1.5")
        self.assertIsNone(code)
        self.assertEqual((self.run_dir() / "transcript.md").read_text(),
                         "# Transcript\n\n")

    def test_diarization_failure_retains_transcript_and_exits_nonzero(self) -> None:
        code, stdout, stderr, _, _, _ = self.invoke(
            failure=RuntimeError("offline diarizer failed")
        )
        self.assertEqual(code, 1)
        self.assertIn("Diarization failed; Whisper transcript retained", stderr)
        self.assertIn("offline diarizer failed", stderr)
        run = self.run_dir()
        self.assertIn(str(run / "transcript.md"), stdout)
        self.assertEqual((run / "transcript.md").read_text(),
                         "# Transcript\n\nHello world!\n")
        self.assertIn(
            "DIARIZATION FAILED", (run / "transcript_speakers.md").read_text()
        )
        self.assertEqual(json.loads((run / "transcript.json").read_text()), [
            {"start": 0.0, "end": 1.0, "text": "Hello", "speaker": None},
            {"start": 1.0, "end": 2.0, "text": "world!", "speaker": None},
        ])

    def test_ordinary_whisper_keeps_legacy_json_and_skips_nemotron(self) -> None:
        with (
            patch("local_transcriber.models.nemotron.installed_nemotron") as installed,
            patch.object(speakers_cli, "_decode") as decode,
            patch.object(cli, "transcribe", return_value={
                "segments": self.segments, "language": "en",
            }) as transcribe,
            redirect_stdout(io.StringIO()),
        ):
            cli.main([
                str(self.source), "--device", "cpu", "--out-dir", str(self.out_dir),
            ])
        installed.assert_not_called()
        decode.assert_not_called()
        transcribe.assert_called_once()
        run = next((self.out_dir / self.source.stem).iterdir())
        self.assertEqual({path.name for path in run.iterdir()}, {
            "transcript.md", "transcript_timestamps.md", "transcript.json",
        })
        self.assertEqual(json.loads((run / "transcript.json").read_text())["engine"],
                         "whisper")

    def test_repeated_diarization_never_replaces_previous_output(self) -> None:
        first_code, _, _, _, _, _ = self.invoke()
        self.assertIsNone(first_code)
        first = self.run_dir()
        before = (first / "transcript.json").read_bytes()
        second_code, _, _, _, _, _ = self.invoke()
        self.assertIsNone(second_code)
        runs = list((self.out_dir / self.source.stem).iterdir())
        self.assertEqual(len(runs), 2)
        self.assertEqual((first / "transcript.json").read_bytes(), before)

    def test_gigaam_rejects_diarize_before_loading_or_creating_output(self) -> None:
        with (
            patch.object(cli, "transcribe_gigaam") as gigaam,
            patch("local_transcriber.models.nemotron.installed_nemotron") as installed,
            redirect_stderr(io.StringIO()) as stderr,
            self.assertRaises(SystemExit) as exit_status,
        ):
            cli.main(self.args("--engine", "gigaam"))
        self.assertEqual(exit_status.exception.code, 2)
        self.assertIn("only with --engine whisper", stderr.getvalue())
        self.assertFalse(self.out_dir.exists())
        gigaam.assert_not_called()
        installed.assert_not_called()

    def test_missing_install_fails_before_asr_and_suggests_install(self) -> None:
        with (
            patch("local_transcriber.models.nemotron.installed_nemotron",
                  side_effect=NemotronInstallError("missing install")),
            patch.object(speakers_cli, "_decode") as decode,
            patch.object(speakers_cli, "_transcribe") as asr,
            patch("local_transcriber.models.nemotron.install_nemotron") as installer,
            redirect_stderr(io.StringIO()) as stderr,
            self.assertRaises(SystemExit) as exit_status,
        ):
            cli.main(self.args())
        self.assertEqual(exit_status.exception.code, 2)
        self.assertIn("just install-nemotron", stderr.getvalue())
        self.assertFalse(self.out_dir.exists())
        decode.assert_not_called()
        asr.assert_not_called()
        installer.assert_not_called()


if __name__ == "__main__":
    unittest.main()
