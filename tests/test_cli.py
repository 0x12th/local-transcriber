from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch


class CliContractTest(unittest.TestCase):
    def run_cli(self, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, "-m", "local_transcriber", *args],
            capture_output=True,
            text=True,
            check=False,
        )

    def test_help_exposes_the_installed_command_contract(self) -> None:
        result = self.run_cli("--help")

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("local-transcriber", result.stdout)
        self.assertIn("--whisper-model", result.stdout)
        self.assertIn("--engine {whisper,gigaam}", result.stdout)
        self.assertIn("--gigaam-model-dir", result.stdout)

    def test_cli_import_does_not_load_engine_dependencies(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                "-S",
                "-c",
                (
                    "import sys; from pathlib import Path; "
                    "import local_transcriber.cli as cli; "
                    "import local_transcriber.models_cli; "
                    "args = cli.parse_args(['install']); "
                    "assert args.input == Path('install'); "
                    "assert args.engine == 'whisper'; "
                    "assert not {'torch', 'whisper', 'onnxruntime', 'numpy', 'yaml', "
                    "'sentencepiece', 'local_transcriber.model_installer', "
                    "'local_transcriber.model_validation'} & sys.modules.keys(); "
                    "cli.main(['--help'])"
                ),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_gigaam_path_does_not_import_whisper_or_torch(self) -> None:
        script = """
import sys
from pathlib import Path
from unittest.mock import patch

import local_transcriber.cli as cli
from local_transcriber.transcript import TranscriptResult

assert not {"torch", "whisper"} & sys.modules.keys()
with (
    patch("local_transcriber.gigaam.GigaAMEngine") as engine_type,
    patch(
        "socket.create_connection",
        side_effect=AssertionError("network must not be used"),
    ) as network,
):
    engine_type.return_value.model_dir = Path("/model")
    engine_type.return_value.transcribe_result.return_value = TranscriptResult(
        [], "ru", 0.0
    )
    cli.transcribe_gigaam(Path("audio.wav"), Path("/model"))
network.assert_not_called()
assert not {
    "torch", "whisper", "local_transcriber.model_installer",
    "local_transcriber.models_cli",
} & sys.modules.keys()
"""
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_rejects_non_positive_speaker_count(self) -> None:
        result = self.run_cli("recording.m4a", "--speaker-count", "0")

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("positive integer", result.stderr)

    def test_invalid_output_root_is_rejected_before_transcription(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "recording.wav"
            source.write_bytes(b"not audio: must not reach Whisper")
            target = root / "recording.json"
            target.write_text("previous result")
            result = self.run_cli(
                str(source), "--device", "cpu", "--out-dir", str(target)
            )
            self.assertEqual(result.returncode, 2)
            self.assertIn("error:", result.stderr)
            self.assertNotIn("Traceback", result.stderr)
            self.assertNotIn("Transcribing", result.stdout)
            self.assertEqual(target.read_text(), "previous result")

    def test_overwrite_option_is_removed(self) -> None:
        result = self.run_cli("recording.wav", "--overwrite")
        self.assertEqual(result.returncode, 2)
        self.assertIn("unrecognized arguments: --overwrite", result.stderr)

    def test_unavailable_device_has_cli_error_before_transcription(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "recording.wav"
            source.write_bytes(b"not audio: must not reach Whisper")
            for device, available in (
                ("cuda", torch.cuda.is_available()),
                ("mps", torch.backends.mps.is_available()),
            ):
                if available:
                    continue
                with self.subTest(device=device):
                    result = self.run_cli(
                        str(source), "--device", device, "--out-dir", directory
                    )
                    self.assertEqual(result.returncode, 2)
                    self.assertIn("not available", result.stderr)
                    self.assertNotIn("Traceback", result.stderr)
                    self.assertNotIn("Transcribing", result.stdout)

    def test_rejects_negative_merge_gap(self) -> None:
        result = self.run_cli("recording.m4a", "--merge-gap-seconds", "-0.1")

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("non-negative number", result.stderr)

    def test_gigaam_rejects_incompatible_options_before_run_directory(self) -> None:
        cases = [
            (["--language", "en"], "supports only --language ru"),
            (["--device", "cuda"], "runs on CPU"),
            (["--device", "mps"], "runs on CPU"),
            (["--initial-prompt", "Terms"], "does not support --initial-prompt"),
            (["--prompt-speakers"], "does not support --initial-prompt"),
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "recording.wav"
            source.write_bytes(b"not audio: must not reach GigaAM")
            for index, (options, message) in enumerate(cases):
                with self.subTest(options=options):
                    out_dir = root / f"out-{index}"
                    result = self.run_cli(
                        str(source),
                        "--engine",
                        "gigaam",
                        "--out-dir",
                        str(out_dir),
                        *options,
                    )
                    self.assertEqual(result.returncode, 2)
                    self.assertIn(message, result.stderr)
                    self.assertNotIn("Traceback", result.stderr)
                    self.assertNotIn("Transcribing", result.stdout)
                    self.assertFalse(out_dir.exists())


if __name__ == "__main__":
    unittest.main()
