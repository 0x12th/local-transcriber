"""Hermetic installer contract tests: no network, source checkout or real build."""

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from local_transcriber.models import nemotron as ni

MODEL = b"pinned model bytes"


class NemotronInstallTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.target = self.root / "model"
        self.real_download = ni._download
        self.real_build = ni._build
        self.real_preflight = ni._preflight
        self.patches = [
            patch.object(ni, "_supported"),
            patch.object(ni, "MODEL_SIZE", len(MODEL)),
            patch.object(ni, "MODEL_SHA256", hashlib.sha256(MODEL).hexdigest()),
            patch.object(ni, "_preflight", side_effect=self.preflight),
            patch.object(ni, "_build", side_effect=self.build),
            patch.object(ni, "_download", side_effect=self.download),
        ]
        for setting in self.patches:
            setting.start()
            self.addCleanup(setting.stop)
        self.calls = []

    def download(self, path):
        self.calls.append("download")
        path.write_bytes(MODEL)

    def build(self, workspace, package):
        self.calls.append("build")
        executable = package / "bin" / "nemo-speech"
        executable.parent.mkdir()
        executable.write_bytes(b"runtime")
        executable.chmod(0o755)

    def preflight(self, executable, model, directory):
        self.calls.append("preflight")
        self.assertEqual(model.read_bytes(), MODEL)
        self.assertEqual(executable.read_bytes(), b"runtime")

    def assert_no_artifacts(self):
        self.assertEqual(list(self.root.iterdir()), [])

    def test_install_and_existing_both_return_model_then_executable_offline(self):
        model, executable = ni.install_nemotron(self.target)
        self.assertEqual(
            (model, executable),
            (self.target / ni.MODEL_NAME, self.target / "bin/nemo-speech"),
        )
        self.assertEqual(self.calls, ["download", "build", "preflight"])
        self.assertEqual(ni.installed_nemotron(self.target), (model, executable))
        self.assertEqual(ni.install_nemotron(self.target), (model, executable))
        self.assertEqual(
            self.calls, ["download", "build", "preflight", "preflight", "preflight"]
        )
        self.assertEqual(sorted(p.name for p in self.root.iterdir()), ["model"])

    def test_preserves_invalid_existing_empty_directory_file_and_symlink(self):
        for kind in ("empty", "file", "symlink"):
            with self.subTest(kind=kind):
                external = self.root / "external"
                external.mkdir(exist_ok=True)
                if kind == "empty":
                    self.target.mkdir()
                elif kind == "file":
                    self.target.write_bytes(b"keep")
                else:
                    self.target.symlink_to(external, target_is_directory=True)
                with self.assertRaisesRegex(
                    ni.NemotronInstallError, "Existing target preserved"
                ):
                    ni.install_nemotron(self.target)
                self.assertEqual(self.calls, [])
                self.assertFalse(list(self.root.glob("*.install-*")))
                if kind == "file":
                    self.assertEqual(self.target.read_bytes(), b"keep")
                if kind == "symlink":
                    self.assertTrue(self.target.is_symlink())
                    self.target.unlink()
                elif kind == "empty":
                    self.target.rmdir()
                else:
                    self.target.unlink()

    def test_corrupt_existing_model_or_manifest_is_never_repaired(self):
        ni.install_nemotron(self.target)
        model = self.target / ni.MODEL_NAME
        model.write_bytes(b"wrong model")
        with self.assertRaises(ni.NemotronInstallError):
            ni.install_nemotron(self.target)
        with self.assertRaises(ni.NemotronInstallError):
            ni.installed_nemotron(self.target)
        self.assertEqual(model.read_bytes(), b"wrong model")
        self.assertEqual(self.calls.count("download"), 1)

    def test_download_failure_does_not_publish_or_leave_lock(self):
        with (
            patch.object(ni, "_download", side_effect=OSError("network down")),
            self.assertRaisesRegex(ni.NemotronInstallError, "network down"),
        ):
            ni.install_nemotron(self.target)
        self.assert_no_artifacts()

    def test_preflight_failure_does_not_publish(self):
        with (
            patch.object(
                ni,
                "_preflight",
                side_effect=ni.NemotronInstallError("V3 pre_ln unsupported"),
            ),
            self.assertRaisesRegex(ni.NemotronInstallError, "pre_ln"),
        ):
            ni.install_nemotron(self.target)
        self.assert_no_artifacts()

    def test_no_replace_race_preserves_foreign_target(self):
        publish = ni._publish_no_replace

        def racing_publish(source, target):
            target.mkdir()
            (target / "foreign").write_bytes(b"do not touch")
            publish(source, target)

        with (
            patch.object(ni, "_publish_no_replace", side_effect=racing_publish),
            self.assertRaises(ni.NemotronInstallError),
        ):
            ni.install_nemotron(self.target)
        self.assertEqual((self.target / "foreign").read_bytes(), b"do not touch")
        self.assertEqual(sorted(p.name for p in self.root.iterdir()), ["model"])

    def test_lock_is_not_stolen_or_cleaned(self):
        lock = self.root / ".model.install-lock"
        lock.mkdir()
        (lock / "foreign").write_bytes(b"keep")
        with self.assertRaisesRegex(ni.NemotronInstallError, "locked"):
            ni.install_nemotron(self.target)
        self.assertEqual((lock / "foreign").read_bytes(), b"keep")
        self.assertEqual(self.calls, [])

    def test_installed_is_offline_and_never_creates_directories(self):
        missing = self.root / "new" / "model"
        with self.assertRaisesRegex(ni.NemotronInstallError, "not installed"):
            ni.installed_nemotron(missing)
        self.assertFalse(missing.parent.exists())
        self.assertEqual(self.calls, [])

    def test_download_bounded_and_verified(self):
        class Response:
            def __init__(self, data, url=ni.MODEL_URL):
                self.data = data
                self.url = url
                self.offset = 0

            def __enter__(self):
                return self

            def __exit__(self, *_):
                pass

            def geturl(self):
                return self.url

            def read(self, count):
                chunk = self.data[self.offset : self.offset + count]
                self.offset += len(chunk)
                return chunk

        with patch.object(ni, "urlopen", return_value=Response(MODEL)):
            self.real_download(self.root / "ok")
        self.assertEqual((self.root / "ok").read_bytes(), MODEL)
        for name, response in (
            ("checksum", Response(b"wrong model bytes")),
            ("oversize", Response(MODEL + b"x")),
            ("redirect", Response(MODEL, "http://example.com/model")),
        ):
            with (
                self.subTest(name=name),
                patch.object(ni, "urlopen", return_value=response),
                self.assertRaises(ni.NemotronInstallError),
            ):
                self.real_download(self.root / name)

    def test_runtime_info_alone_does_not_skip_metal_inference(self):
        executable = self.root / "nemo-speech"
        executable.write_bytes(b"runtime")
        executable.chmod(0o755)
        model = self.root / "pinned.gguf"
        model.write_bytes(MODEL)
        commands = []

        def runner(command, cwd, timeout):
            commands.append(command)
            if command[1:3] == ["model", "info"]:
                return json.dumps(
                    {
                        "role": "diarization",
                        "name": "Nemotron-3-Diarization.q8_0",
                        "runtime_compatible": True,
                    }
                )
            if command[1] == "diarize":
                self.assertTrue(Path(command[2]).is_file())
                self.assertEqual(command[command.index("--device") + 1], "metal")
                raise ni.NemotronInstallError("unsupported pre_ln")
            return "nemo-speech 0.1.0"

        with (
            patch.object(ni, "_run", side_effect=runner),
            self.assertRaisesRegex(ni.NemotronInstallError, "pre_ln"),
        ):
            self.real_preflight(executable, model, self.root)
        self.assertEqual(
            [command[1] for command in commands], ["--version", "model", "diarize"]
        )
        self.assertEqual(
            sorted(p.name for p in self.root.iterdir()), ["nemo-speech", "pinned.gguf"]
        )

    def test_unsupported_platform_fails_before_creating_target_or_network(self):
        with (
            patch.object(
                ni,
                "_supported",
                side_effect=ni.NemotronInstallError("Metal unsupported"),
            ),
            self.assertRaisesRegex(ni.NemotronInstallError, "Metal unsupported"),
        ):
            ni.install_nemotron(self.target)
        self.assert_no_artifacts()
        self.assertEqual(self.calls, [])

    def test_wrong_source_commit_fails_before_configure(self):
        workspace = self.root / "workspace"
        workspace.mkdir()
        package = workspace / "package"
        package.mkdir()
        commands = []

        def runner(command, cwd, timeout):
            commands.append(command)
            return "wrong commit"

        with (
            patch.object(ni, "_run", side_effect=runner),
            self.assertRaisesRegex(ni.NemotronInstallError, "pinned commit"),
        ):
            self.real_build(workspace, package)
        self.assertEqual(len(commands), 3)
        self.assertFalse(any(cmd[0] == "cmake" for cmd in commands))

    def test_run_timeout_reports_command_cwd_and_both_partial_streams(self):
        # Mimics the timed-out upstream configure step without network or a build.
        command = [
            sys.executable,
            "-u",
            "-c",
            "import sys, time; print('patch stage', flush=True); "
            "print('cmake diagnostic', file=sys.stderr, flush=True); time.sleep(10)",
        ]
        with self.assertRaises(ni.NemotronInstallError) as caught:
            ni._run(command, self.root, 1)
        message = str(caught.exception)
        for evidence in (
            "timed out",
            "1s",
            str(self.root),
            "patch stage",
            "cmake diagnostic",
        ):
            with self.subTest(evidence=evidence):
                self.assertIn(evidence, message)

    def test_compiler_normal_header_probe_never_uses_sdk_fallback(self):
        with (
            patch.dict("os.environ", {"CXX": "c++", "CXXFLAGS": ""}),
            patch.object(
                ni.subprocess,
                "run",
                return_value=subprocess.CompletedProcess(["c++"], 0, "", ""),
            ) as run,
            patch.object(ni, "_run", side_effect=AssertionError("unexpected xcrun")),
        ):
            self.assertEqual(ni._cxx_configure_flags(self.root), [])
        self.assertEqual(run.call_count, 1)
        self.assertIn("#include <array>", run.call_args.kwargs["input"])

    def test_compiler_sdk_fallback_requires_header_and_successful_probe(self):
        sdk = self.root / "SDK with spaces"
        header = sdk / "usr/include/c++/v1/array"
        header.parent.mkdir(parents=True)
        header.write_text("header", encoding="utf-8")
        calls = []

        def compile_probe(command, **kwargs):
            calls.append(command)
            return subprocess.CompletedProcess(
                command,
                0 if "-isystem" in command else 1,
                "",
                "" if "-isystem" in command else "fatal error: 'array' file not found",
            )

        with (
            patch.dict("os.environ", {"CXX": "c++", "CXXFLAGS": "-O2"}),
            patch.object(ni.subprocess, "run", side_effect=compile_probe),
            patch.object(ni, "_run", return_value=str(sdk)) as sdk_lookup,
        ):
            flags = ni._cxx_configure_flags(self.root)
        self.assertEqual(
            flags, [f"-DCMAKE_CXX_FLAGS:STRING=-O2 -isystem '{header.parent}'"]
        )
        self.assertEqual(
            sdk_lookup.call_args.args[0],
            ["xcrun", "--sdk", "macosx", "--show-sdk-path"],
        )
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[1][calls[1].index("-isystem") + 1], str(header.parent))

    def test_compiler_sdk_fallback_rejects_missing_or_unusable_header(self):
        sdk = self.root / "sdk"
        header = sdk / "usr/include/c++/v1/array"
        with (
            patch.dict("os.environ", {"CXX": "c++", "CXXFLAGS": ""}),
            patch.object(
                ni.subprocess,
                "run",
                return_value=subprocess.CompletedProcess(
                    ["c++"], 1, "", "fatal error: 'array' file not found"
                ),
            ) as run,
            patch.object(ni, "_run", return_value=str(sdk)),
            self.assertRaisesRegex(ni.NemotronInstallError, r"SDK libc\+\+ header"),
        ):
            ni._cxx_configure_flags(self.root)
        self.assertEqual(run.call_count, 1)
        header.parent.mkdir(parents=True)
        header.write_text("header", encoding="utf-8")
        with (
            patch.dict("os.environ", {"CXX": "c++", "CXXFLAGS": ""}),
            patch.object(
                ni.subprocess,
                "run",
                return_value=subprocess.CompletedProcess(
                    ["c++"], 1, "", "fatal error: 'array' file not found"
                ),
            ) as run,
            patch.object(ni, "_run", return_value=str(sdk)),
            self.assertRaisesRegex(
                ni.NemotronInstallError, "SDK header probe failed.*array"
            ),
        ):
            ni._cxx_configure_flags(self.root)
        self.assertEqual(run.call_count, 2)

    def test_source_checkout_and_build_are_pinned(self):
        workspace = self.root / "workspace"
        workspace.mkdir()
        package = workspace / "package"
        package.mkdir()
        commands = []

        def runner(command, cwd, timeout):
            commands.append(command)
            if command[:3] == ["git", "-C", "ggml"]:
                return ni.GGML_COMMIT
            if command[-2:] == ["rev-parse", "HEAD"]:
                return ni.SOURCE_COMMIT
            return ""

        with (
            patch.object(ni, "_run", side_effect=runner),
            patch.object(
                ni,
                "_cxx_configure_flags",
                return_value=["-DCMAKE_CXX_FLAGS:STRING=-isystem /sdk/include"],
            ),
        ):
            self.real_build(workspace, package)
        self.assertIn(["git", "checkout", "--detach", ni.SOURCE_COMMIT], commands)
        self.assertIn(
            [
                "bash",
                "scripts/configure.sh",
                "metal-diar",
                "-DGGML_NATIVE=OFF",
                "-DCMAKE_CXX_FLAGS:STRING=-isystem /sdk/include",
            ],
            commands,
        )
        self.assertIn(
            ["cmake", "--build", "--preset", "metal-diar", "--parallel", "4"], commands
        )
        self.assertTrue(
            any(
                cmd[:3] == ["cmake", "--install", "build/metal-diar"]
                for cmd in commands
            )
        )


if __name__ == "__main__":
    unittest.main()
