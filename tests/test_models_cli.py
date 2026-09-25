"""Model CLI contracts with synthetic bundles only; never real downloads/ASR."""

from __future__ import annotations

import hashlib
import io
import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import yaml

from local_transcriber import model_installer as mi
from local_transcriber import model_validation as mv
from local_transcriber import models_cli
from tests.test_model_installer import FakeResponse, archive_bytes, member
from tests.test_model_validation import CONFIG, FakeRuntime


class ModelsCliTest(unittest.TestCase):
    def invoke(self, *args: str) -> tuple[int, str, str]:
        stdout, stderr = io.StringIO(), io.StringIO()
        code = 0
        with redirect_stdout(stdout), redirect_stderr(stderr):
            try:
                models_cli.main(list(args))
            except SystemExit as error:
                assert isinstance(error.code, int)
                code = error.code
        return code, stdout.getvalue(), stderr.getvalue()

    def test_help_and_parse_errors_are_stdlib_only_without_installer(self):
        # Fresh -S processes have no site-packages, even in the extra-enabled env.
        script = """
import sys
from importlib.abc import MetaPathFinder
class Guard(MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if (fullname.split('.')[0] in {
            'torch', 'whisper', 'numpy', 'onnxruntime', 'sentencepiece', 'yaml'
        } or fullname in {
            'local_transcriber.model_installer', 'local_transcriber.model_validation',
            'local_transcriber.gigaam', 'local_transcriber.cli'
        }):
            raise AssertionError('Unexpected import: ' + fullname)
sys.meta_path.insert(0, Guard())
def audit(event, args):
    if event.startswith('socket.') or event in {'subprocess.Popen', 'os.system'}:
        raise AssertionError('Unexpected external operation: ' + event)
sys.addaudithook(audit)
from local_transcriber.models_cli import main
main(sys.argv[1:])
"""
        cases = [
            (["--help"], 0, "install"),
            (["install", "--help"], 0, "--model-dir"),
            (["install", "gigaam", "--help"], 0, "managed cache"),
            ([], 2, "required"),
            (["install"], 2, "required"),
            (["remove", "gigaam"], 2, "invalid choice"),
            (["install", "whisper"], 2, "invalid choice"),
            (["install", "gigaam", "--model-dir"], 2, "expected one argument"),
            (["install", "gigaam", "--force"], 2, "unrecognized arguments"),
        ]
        for args, expected_code, message in cases:
            with self.subTest(args=args):
                result = subprocess.run(
                    [sys.executable, "-S", "-c", script, *args],
                    capture_output=True, text=True, check=False, timeout=15,
                )
                self.assertEqual(result.returncode, expected_code, result.stderr)
                self.assertIn(message, result.stdout if expected_code == 0
                              else result.stderr)
                self.assertNotIn("Traceback", result.stderr)

    def test_preflight_opts_out_before_first_runtime_import(self):
        script = """
import sys
from pathlib import Path
from unittest.mock import patch
from local_transcriber import model_installer as mi, model_validation as mv
from local_transcriber.models_cli import main
from tests.test_model_validation import FakeRuntime, fresh_ort_import
missing = sys.argv[1] == 'missing'
runtime = FakeRuntime()
metadata = mv.ModelMetadata(Path('/synthetic-model'), (), 'pinned', True, ())
with (
    fresh_ort_import(runtime, missing=missing) as observed,
    patch.object(mi, 'install_model', return_value=mi.InstallResult(
        metadata.model_dir, metadata, True
    )) as install,
    patch.object(mi, 'urlopen', side_effect=AssertionError('No download')),
    patch.object(mv, 'load_model_directory',
                 side_effect=AssertionError('Duplicate model loading')),
):
    try:
        main(['install', 'gigaam'])
        assert not missing
    except SystemExit as error:
        assert missing and error.code == 1
assert observed == ['1']
if missing:
    install.assert_not_called()
else:
    install.assert_called_once_with(None)
assert runtime.session_calls == []
assert runtime.tokenizer_paths == []
"""
        for inherited in (None, "0"):
            for runtime in ("available", "missing"):
                with self.subTest(inherited=inherited, runtime=runtime):
                    env = dict(os.environ)
                    env.pop("ORT_DISABLE_TELEMETRY", None)
                    if inherited is not None:
                        env["ORT_DISABLE_TELEMETRY"] = inherited
                    result = subprocess.run(
                        [sys.executable, "-B", "-c", script, runtime], env=env,
                        capture_output=True, text=True, check=False, timeout=15,
                    )
                    self.assertEqual(result.returncode, 0, result.stderr)
                    if runtime == "missing":
                        self.assertIn("onnxruntime", result.stderr)
                        self.assertIn("uv sync --locked --extra gigaam", result.stderr)
                    else:
                        self.assertIn("Already installed:", result.stdout)

    def test_import_help_and_whisper_do_not_initialize_ort_or_change_policy(self):
        script = """
import os, sys, tempfile
from importlib.abc import MetaPathFinder
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
class Guard(MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'onnxruntime', 'sentencepiece', 'numpy', 'yaml'}:
            raise AssertionError('Unexpected runtime import: ' + fullname)
sys.meta_path.insert(0, Guard())
import local_transcriber
from local_transcriber import cli, model_validation, models_cli
for main in (cli.main, models_cli.main):
    try:
        main(['--help'])
    except SystemExit as error:
        assert error.code == 0
result = {'segments': [], 'language': 'ru', 'duration': 0.0}
model = SimpleNamespace(transcribe=Mock(return_value=result))
whisper = SimpleNamespace(load_model=Mock(return_value=model))
with tempfile.TemporaryDirectory() as directory, patch.dict(sys.modules, {
    'whisper': whisper, 'torch': SimpleNamespace(),
}):
    root = Path(directory)
    source = root / 'synthetic.wav'
    source.write_bytes(b'not audio; fake Whisper boundary')
    cli.main([str(source), '--device', 'cpu', '--out-dir', str(root / 'out')])
whisper.load_model.assert_called_once_with('turbo', device='cpu')
model.transcribe.assert_called_once()
assert 'onnxruntime' not in sys.modules
assert os.environ['ORT_DISABLE_TELEMETRY'] == '0'
"""
        result = subprocess.run(
            [sys.executable, "-B", "-S", "-c", script],
            env={**os.environ, "ORT_DISABLE_TELEMETRY": "0"},
            capture_output=True, text=True, check=False, timeout=15,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Transcribing with Whisper", result.stdout)

    def test_parser_install_target(self):
        parser = models_cli.build_parser()
        default = parser.parse_args(["install", "gigaam"])
        self.assertEqual((default.command, default.model, default.model_dir),
                         ("install", "gigaam", None))
        explicit = parser.parse_args(["install", "gigaam", "--model-dir", "~/my model"])
        self.assertEqual(explicit.model_dir, Path("~/my model"))

    def test_target_delegation_and_pinned_success_messages(self):
        # Installer is the resource boundary: the CLI must call it exactly once,
        # without resolving env/legacy paths or loading another model itself.
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory).resolve() / "actual-model"
            metadata = mv.ModelMetadata(target, (), "pinned", True, ())
            targets = (([], None),
                       (["--model-dir", "~/my model"], Path("~/my model")))
            for options, expected in targets:
                for already_installed in (False, True):
                    with (
                        self.subTest(options=options, already=already_installed),
                        patch.dict(os.environ, {"GIGAAM_MODEL_DIR": "/ignored-env"}),
                        patch.object(mi, "install_model", return_value=mi.InstallResult(
                            target, metadata, already_installed
                        )) as install,
                        patch.object(mv, "resolve_model_dir",
                                     side_effect=AssertionError("ASR resolver used")),
                        patch.object(mv, "load_model_directory",
                                     side_effect=AssertionError("Duplicate loading")),
                    ):
                        code, out, err = self.invoke("install", "gigaam", *options)
                        self.assertEqual((code, err), (0, ""))
                        self.assertIn(str(target), out)
                        self.assertIn("Profile: v3_e2e_rnnt; verification: pinned", out)
                        self.assertIn("Already installed:" if already_installed
                                      else "Installed:", out)
                        install.assert_called_once_with(expected)

    def test_missing_or_broken_extra_fails_before_install_with_setup_instruction(self):
        errors: list[ImportError | OSError] = [
            ModuleNotFoundError(f"No module named '{name}'", name=name)
            for name in ("numpy", "onnxruntime", "sentencepiece", "yaml")
        ]
        errors += [ImportError("transitive dependency is broken"),
                   OSError("native library could not be loaded")]
        for error in errors:
            with (
                self.subTest(error=error),
                patch.object(models_cli.importlib, "import_module", side_effect=error),
                patch.object(mi, "install_model") as install,
            ):
                code, out, err = self.invoke("install", "gigaam")
                self.assertEqual((code, out), (1, ""))
                self.assertIn(str(error), err)
                self.assertIn("uv sync --locked --extra gigaam", err)
                self.assertNotIn("Traceback", err)
                install.assert_not_called()

    def test_install_preflight_does_not_import_transcription_engines(self):
        script = """
import sys
from importlib.abc import MetaPathFinder
from pathlib import Path
from unittest.mock import patch
class Guard(MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if (fullname.split('.')[0] in {'torch', 'whisper'} or fullname in {
            'local_transcriber.gigaam', 'local_transcriber.cli'
        }):
            raise AssertionError('Unexpected engine import: ' + fullname)
sys.meta_path.insert(0, Guard())
from local_transcriber import model_installer as mi, model_validation as mv
from local_transcriber.models_cli import main
metadata = mv.ModelMetadata(Path('/synthetic-model'), (), 'pinned', False, ())
with patch.object(mi, 'install_model', return_value=mi.InstallResult(
    metadata.model_dir, metadata, True
)) as install:
    main(['install', 'gigaam'])
install.assert_called_once_with(None)
"""
        result = subprocess.run(
            [sys.executable, "-c", script], capture_output=True,
            text=True, check=False, timeout=15,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Already installed: /synthetic-model", result.stdout)
        self.assertIn("verification: pinned", result.stdout)

    def test_base_environment_install_reports_missing_extra_without_traceback(self):
        result = subprocess.run(
            [sys.executable, "-S", "-m", "local_transcriber.models_cli",
             "install", "gigaam"],
            capture_output=True, text=True, check=False, timeout=15,
        )
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn("uv sync --locked --extra gigaam", result.stderr)
        self.assertNotIn("Traceback", result.stderr)
        self.assertEqual(result.stdout, "")

    def test_installer_failures_keep_reason_and_nonzero_exit_without_retry(self):
        reasons = [
            "GigaAM installation failed: network connection timed out",
            "Model archive SHA-256 does not match pinned checksum",
            "Unexpected archive entry: other-file",
            "Permission denied: target parent",
            "Model target is locked: target.install-lock",
            "Existing target preserved; choose an absent target path: manual-model",
            "Cleanup failed after publication: staging",
        ]
        for reason in reasons:
            with (
                self.subTest(reason=reason),
                patch.object(mi, "install_model",
                             side_effect=mi.ModelInstallError(reason)) as install,
            ):
                code, out, err = self.invoke(
                    "install", "gigaam", "--model-dir", "target"
                )
                self.assertEqual((code, out), (1, ""))
                self.assertIn(reason, err)
                self.assertNotIn("Traceback", err)
                install.assert_called_once_with(Path("target"))

    def test_real_installer_through_cli_install_and_repeat_with_fake_bundle(self):
        contents = {f.name: f"synthetic {f.name}".encode() for f in mv.BUNDLE.files}
        contents["v3_e2e_rnnt.yaml"] = yaml.safe_dump(CONFIG).encode()
        data = archive_bytes([member(f"gigaam-v3-onnx-int8/{name}", value)
                              for name, value in contents.items()])
        spec = replace(
            mv.BUNDLE, archive_size_bytes=len(data),
            archive_sha256=hashlib.sha256(data).hexdigest(),
            files=tuple(mv.FileSpec(name, len(value), hashlib.sha256(value).hexdigest())
                        for name, value in contents.items()),
        )
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(mv, "BUNDLE", spec),
            FakeRuntime().installed() as runtime,
            patch.object(mi, "urlopen",
                         return_value=FakeResponse(data, [])) as transport,
        ):
            target = Path(directory).resolve() / "installed"
            args = ("install", "gigaam", "--model-dir", str(target))
            code, out, err = self.invoke(*args)
            self.assertEqual((code, err), (0, ""))
            self.assertIn(f"Installed: {target}", out)
            self.assertIn("verification: pinned", out)
            self.assertEqual(len(runtime.session_calls), 3)
            self.assertEqual(len(runtime.tokenizer_paths), 1)
            before = {p.name: (p.read_bytes(), p.stat().st_mtime_ns)
                      for p in target.iterdir()}
            code, out, err = self.invoke(*args)
            self.assertEqual((code, err), (0, ""))
            self.assertIn(f"Already installed: {target}", out)
            self.assertEqual(len(runtime.session_calls), 6)
            self.assertEqual(len(runtime.tokenizer_paths), 2)
            transport.assert_called_once()
            self.assertEqual(before, {p.name: (p.read_bytes(), p.stat().st_mtime_ns)
                                      for p in target.iterdir()})
            self.assertEqual(set(before), set(contents) | {"model-manifest.json"})
            self.assertEqual(list(target.parent.iterdir()), [target])
            self.assertFalse(any(runtime.runs.values()))
            self.assertEqual(runtime.decoded, [])

    def test_manual_compatible_target_is_preserved_with_absent_target_instruction(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            FakeRuntime().installed(),
            patch.object(mi, "urlopen", side_effect=AssertionError("No download")),
        ):
            target = Path(directory).resolve()
            for file in mv.BUNDLE.files:
                (target / file.name).write_bytes(b"synthetic manual weights")
            (target / "v3_e2e_rnnt.yaml").write_text(yaml.safe_dump(CONFIG))
            before = {p.name: p.read_bytes() for p in target.iterdir()}
            code, out, err = self.invoke(
                "install", "gigaam", "--model-dir", str(target)
            )
            self.assertEqual((code, out), (1, ""))
            self.assertIn(
                "Existing target preserved; choose an absent target path", err
            )
            self.assertIn("not the exact pinned bundle", err)
            self.assertNotIn("Traceback", err)
            self.assertEqual(before, {p.name: p.read_bytes() for p in target.iterdir()})


if __name__ == "__main__":
    unittest.main()
