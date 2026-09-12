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


if __name__ == "__main__":
    unittest.main()
