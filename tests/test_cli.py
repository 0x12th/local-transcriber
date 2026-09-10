from __future__ import annotations

import subprocess
import sys
import unittest


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

    def test_rejects_negative_merge_gap(self) -> None:
        result = self.run_cli("recording.m4a", "--merge-gap-seconds", "-0.1")

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("non-negative number", result.stderr)


if __name__ == "__main__":
    unittest.main()
