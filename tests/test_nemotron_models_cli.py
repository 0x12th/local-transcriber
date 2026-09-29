"""The Nemotron setup command remains explicit and independent of GigaAM."""

from __future__ import annotations

import io
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from local_transcriber import models_cli
from local_transcriber.models import nemotron


class NemotronModelsCliTest(unittest.TestCase):
    def test_install_dispatch_needs_no_gigaam_dependencies(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "managed"
            model = target / nemotron.MODEL_NAME
            runtime = target / "bin" / "nemo-speech"
            output = io.StringIO()
            with (
                patch.object(
                    nemotron, "install_nemotron", return_value=(model, runtime)
                ) as install,
                patch.object(
                    models_cli.importlib, "import_module",
                    side_effect=AssertionError("GigaAM import on Nemotron install"),
                ),
                redirect_stdout(output),
            ):
                models_cli.main(["install", "nemotron", "--model-dir", str(target)])
            install.assert_called_once_with(target)
            self.assertIn(str(model), output.getvalue())
            self.assertIn(str(runtime), output.getvalue())

    def test_installer_error_is_reported_without_traceback(self) -> None:
        with (
            patch.object(
                nemotron,
                "install_nemotron",
                side_effect=nemotron.NemotronInstallError("Metal unavailable"),
            ),
            redirect_stderr(io.StringIO()) as stderr,
            self.assertRaises(SystemExit) as exit_status,
        ):
            models_cli.main(["install", "nemotron"])
        self.assertEqual(exit_status.exception.code, 1)
        self.assertIn("Metal unavailable", stderr.getvalue())
