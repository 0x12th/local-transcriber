"""Explicit, no-replace Nemotron 3 diarization installation for macOS arm64.

Model: NVIDIA Open Model License (OpenMDW 1.1), https://huggingface.co/nvidia/Nemotron-3-Diarization
Runtime: Apache-2.0 plus third-party notices, https://github.com/NVIDIA/NeMo-Speech.cpp
Neither importing this module nor installed_nemotron downloads anything. The
installer builds the pinned source (not the incompatible v0.1.0 release).
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shlex
import shutil
import stat
import subprocess
import sys
import tempfile
import time
import wave
from pathlib import Path
from urllib.request import urlopen

from local_transcriber.models.installer import _identity, _publish_no_replace

SOURCE_URL = "https://github.com/NVIDIA/NeMo-Speech.cpp.git"
SOURCE_COMMIT = "97a15afa5caa9bce5baaa86c1184103877af4101"
GGML_COMMIT = "c03b4e2bcece5134827881af90242086daf75be5"
MODEL_REVISION = "f667ed73aee57d40cc39428eb768b4fd87a0a29e"
MODEL_NAME = "Nemotron-3-Diarization.q8_0.gguf"
MODEL_URL = (
    "https://huggingface.co/nvidia/Nemotron-3-Diarization/resolve/"
    f"{MODEL_REVISION}/{MODEL_NAME}"
)
MODEL_SIZE = 107012128
MODEL_SHA256 = "08456d9e22cd9a323c0364d98375f3746d6e68507ebb705cd46438c534c7a3a1"
CACHE_NAME = "nemotron-3-diarization-q8_0-metal"
MANIFEST_NAME = "install.json"
BLOCK_BYTES = 1024 * 1024


class NemotronInstallError(RuntimeError):
    """The existing target was preserved or the new installation failed."""


def _target(model_dir: Path | None) -> Path:
    requested = (
        model_dir
        if model_dir is not None
        else Path(os.environ.get("XDG_CACHE_HOME") or "~/.cache").expanduser()
        / "local-transcriber"
        / "models"
        / CACHE_NAME
    ).expanduser()
    # Resolve parents but not the final component (which may be a symlink).
    return requested.parent.resolve() / requested.name


def _supported() -> None:
    if sys.platform != "darwin" or platform.machine() != "arm64":
        raise NemotronInstallError(
            "Nemotron Metal install requires macOS Apple Silicon"
        )


def _digest(path: Path) -> tuple[int, str]:
    sha = hashlib.sha256()
    count = 0
    with path.open("rb") as stream:
        while block := stream.read(BLOCK_BYTES):
            count += len(block)
            if count > MODEL_SIZE:
                raise NemotronInstallError("GGUF exceeds pinned size")
            sha.update(block)
    return count, sha.hexdigest()


def _check_model(path: Path) -> None:
    if not stat.S_ISREG(path.lstat().st_mode) or _digest(path) != (
        MODEL_SIZE,
        MODEL_SHA256,
    ):
        raise NemotronInstallError("GGUF is not the pinned Nemotron 3 model")


def _download(path: Path) -> None:
    deadline = time.monotonic() + 1800
    count = 0
    sha = hashlib.sha256()
    with urlopen(MODEL_URL, timeout=30) as response, path.open("xb") as output:
        if response.geturl().split(":", 1)[0] != "https":
            raise NemotronInstallError("GGUF download redirected outside HTTPS")
        while True:
            if time.monotonic() > deadline:
                raise NemotronInstallError("GGUF download exceeded 30 minutes")
            block = response.read(min(BLOCK_BYTES, MODEL_SIZE - count + 1))
            if not block:
                break
            count += len(block)
            if count > MODEL_SIZE:
                raise NemotronInstallError("GGUF exceeds pinned size")
            sha.update(block)
            output.write(block)
    if (count, sha.hexdigest()) != (MODEL_SIZE, MODEL_SHA256):
        raise NemotronInstallError("GGUF size or SHA-256 does not match pinned model")


def _run(command: list[str], cwd: Path, timeout: int) -> str:
    def excerpt(output: str | bytes | None) -> str:
        # TimeoutExpired can contain bytes even when text=True.
        if isinstance(output, bytes):
            output = output.decode("utf-8", errors="replace")
        return (output or "").strip()[-1200:] or "(empty)"

    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        raise NemotronInstallError(
            f"Command {command!r} timed out after {timeout}s in {cwd}; "
            f"stdout (last 1200 chars): {excerpt(error.stdout)}; "
            f"stderr (last 1200 chars): {excerpt(error.stderr)}"
        ) from error
    if result.returncode:
        raise NemotronInstallError(
            f"Command {command!r} failed ({result.returncode}) in {cwd}; "
            f"stdout (last 1200 chars): {excerpt(result.stdout)}; "
            f"stderr (last 1200 chars): {excerpt(result.stderr)}"
        )
    return result.stdout.strip()


def _cxx_configure_flags(cwd: Path) -> list[str]:
    """Work around CLT layouts that omit libc++ from the compiler search path."""
    compiler = shlex.split(os.environ.get("CXX") or "c++")
    flags = shlex.split(os.environ.get("CXXFLAGS", ""))
    source = "#include <array>\nint main() { std::array<int, 1> a{}; return a[0]; }\n"

    def probe(extra: list[str]) -> subprocess.CompletedProcess[str]:
        try:
            return subprocess.run(
                [
                    *compiler,
                    *flags,
                    *extra,
                    "-x",
                    "c++",
                    "-std=c++17",
                    "-fsyntax-only",
                    "-",
                ],
                input=source,
                cwd=cwd,
                capture_output=True,
                text=True,
                check=False,
                timeout=15,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise NemotronInstallError(
                f"C++ <array> compiler probe failed: {error}"
            ) from error

    normal = probe([])
    if normal.returncode == 0:
        return []
    try:
        sdk = Path(_run(["xcrun", "--sdk", "macosx", "--show-sdk-path"], cwd, 15))
    except (OSError, NemotronInstallError) as error:
        raise NemotronInstallError(
            f"C++ <array> probe failed: {normal.stderr.strip()}; "
            f"cannot locate macOS SDK: {error}"
        ) from error
    include = sdk / "usr" / "include" / "c++" / "v1"
    if not (include / "array").is_file():
        raise NemotronInstallError(
            f"C++ <array> probe failed: {normal.stderr.strip()}; "
            f"SDK libc++ header not found: {include / 'array'}"
        )
    fallback = probe(["-isystem", str(include)])
    if fallback.returncode:
        raise NemotronInstallError(
            f"C++ <array> probe failed: {normal.stderr.strip()}; "
            f"SDK header probe failed: {fallback.stderr.strip()}"
        )
    return [
        f"-DCMAKE_CXX_FLAGS:STRING={shlex.join([*flags, '-isystem', str(include)])}"
    ]


def _build(workspace: Path, package: Path) -> None:
    source = workspace / "source"
    # The clone and all build outputs live in private staging, never in the user's
    # checkout. The upstream helper applies the pinned Metal ggml patch series.
    _run(
        [
            "git",
            "clone",
            "--no-checkout",
            "--filter=blob:none",
            SOURCE_URL,
            str(source),
        ],
        workspace,
        900,
    )
    _run(["git", "checkout", "--detach", SOURCE_COMMIT], source, 300)
    if _run(["git", "rev-parse", "HEAD"], source, 30) != SOURCE_COMMIT:
        raise NemotronInstallError("Source checkout did not match pinned commit")
    _run(["git", "submodule", "update", "--init", "--depth", "1", "ggml"], source, 900)
    if _run(["git", "-C", "ggml", "rev-parse", "HEAD"], source, 30) != GGML_COMMIT:
        raise NemotronInstallError("ggml checkout did not match pinned submodule")
    # ggml's native ARM run-probes stall at i8mm on M1; the working pilot
    # configures GGML_NATIVE=OFF while keeping GGML_METAL=ON via the preset.
    _run(
        [
            "bash",
            "scripts/configure.sh",
            "metal-diar",
            "-DGGML_NATIVE=OFF",
            *_cxx_configure_flags(source),
        ],
        source,
        300,
    )
    _run(
        ["cmake", "--build", "--preset", "metal-diar", "--parallel", "4"], source, 1800
    )
    _run(
        ["cmake", "--install", "build/metal-diar", "--prefix", str(package)],
        source,
        300,
    )


def _preflight(executable: Path, model: Path, directory: Path) -> None:
    if not stat.S_ISREG(executable.lstat().st_mode) or not os.access(
        executable, os.X_OK
    ):
        raise NemotronInstallError("Built nemo-speech is not executable")
    _run([str(executable), "--version"], directory, 15)
    details = json.loads(
        _run([str(executable), "model", "info", str(model)], directory, 30)
    )
    if (
        details.get("role") != "diarization"
        or "nemotron-3-diarization" not in details.get("name", "").lower()
        or details.get("runtime_compatible") is not True
    ):
        raise NemotronInstallError("Runtime cannot load pinned Nemotron 3 GGUF")
    # model info alone did not catch the v0.1.0 V3 pre_ln incompatibility.
    with tempfile.TemporaryDirectory(prefix="nemotron-preflight-") as temp:
        silent = Path(temp) / "silence.wav"
        with wave.open(str(silent), "wb") as audio:
            audio.setnchannels(1)
            audio.setsampwidth(2)
            audio.setframerate(16000)
            audio.writeframes(b"\0" * 32000)
        _run(
            [
                str(executable),
                "diarize",
                str(silent),
                "--model",
                str(model),
                "--device",
                "metal",
                "--format",
                "rttm",
                "--output",
                str(Path(temp) / "silence.rttm"),
            ],
            directory,
            90,
        )


def _paths(target: Path) -> tuple[Path, Path]:
    return target / MODEL_NAME, target / "bin" / "nemo-speech"


def _validate(target: Path) -> tuple[Path, Path]:
    if not stat.S_ISDIR(target.lstat().st_mode):
        raise NemotronInstallError("Target is not a regular directory")
    manifest_path = target / MANIFEST_NAME
    if not stat.S_ISREG(manifest_path.lstat().st_mode):
        raise NemotronInstallError("Missing install manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest != {
        "source_commit": SOURCE_COMMIT,
        "model_revision": MODEL_REVISION,
        "model_sha256": MODEL_SHA256,
        "preset": "metal-diar",
    }:
        raise NemotronInstallError("Existing install has unknown provenance")
    model, executable = _paths(target)
    _check_model(model)
    _preflight(executable, model, target)
    return model, executable


def installed_nemotron(model_dir: Path | None = None) -> tuple[Path, Path]:
    """Return verified (GGUF, executable) from an existing install; offline/read-only.

    This does a short Metal inference, not just a model-info check. Raises
    NemotronInstallError on missing/incompatible installs; never falls back.
    """
    _supported()
    target = _target(model_dir)
    try:
        return _validate(target)
    except (OSError, ValueError, RuntimeError) as error:
        raise NemotronInstallError(
            f"Nemotron is not installed at {target}: {error}"
        ) from error


def install_nemotron(model_dir: Path | None = None) -> tuple[Path, Path]:
    """Build and install on explicit request only; never replace an existing target.

    Existing verified targets are offline idempotent successes. A crash may leave
    an install lock; it must be inspected and removed manually. The target parent
    must be user-controlled (not a sandbox against a malicious local actor).
    """
    _supported()
    target = _target(model_dir)
    workspace: Path | None = None
    lock: Path | None = None
    lock_identity: tuple[int, int] | None = None
    workspace_identity: tuple[int, int] | None = None
    result: tuple[Path, Path] | None = None
    failure: Exception | None = None
    cleanup_errors: list[str] = []
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        lock = target.with_name(f".{target.name}.install-lock")
        try:
            lock.mkdir(mode=0o700)
        except FileExistsError:
            raise NemotronInstallError(
                f"Target is locked: {lock}; no stale-lock removal"
            ) from None
        lock_identity = _identity(lock)
        try:
            previous = target.lstat()
        except FileNotFoundError:
            previous = None
        if previous is not None:
            try:
                result = _validate(target)
                if _identity(target) != (previous.st_dev, previous.st_ino):
                    raise NemotronInstallError("Target changed during validation")
            except (OSError, ValueError, RuntimeError) as error:
                raise NemotronInstallError(
                    f"Existing target preserved: {target}: {error}; "
                    "choose an absent path"
                ) from error
        else:
            workspace = Path(
                tempfile.mkdtemp(prefix=f".{target.name}.install-", dir=target.parent)
            )
            workspace_identity = _identity(workspace)
            package = workspace / "package"
            package.mkdir()
            model, executable = _paths(package)
            print("Downloading Nemotron 3 GGUF...", file=sys.stderr)
            _download(model)
            _check_model(model)
            print("Building pinned nemo-speech Metal runtime...", file=sys.stderr)
            _build(workspace, package)
            print("Verifying Metal inference...", file=sys.stderr)
            _preflight(executable, model, workspace)
            (package / MANIFEST_NAME).write_text(
                json.dumps(
                    {
                        "source_commit": SOURCE_COMMIT,
                        "model_revision": MODEL_REVISION,
                        "model_sha256": MODEL_SHA256,
                        "preset": "metal-diar",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            _publish_no_replace(package, target)
            result = _paths(target)
    except Exception as error:
        failure = error
    finally:
        for path, identity, recursive in (
            (workspace, workspace_identity, True),
            (lock, lock_identity, False),
        ):
            if path is not None and identity is not None:
                try:
                    if _identity(path) != identity:
                        raise NemotronInstallError(
                            f"Owned path replaced; left untouched: {path}"
                        )
                    if recursive:
                        shutil.rmtree(path)
                    else:
                        path.rmdir()
                except Exception as error:
                    cleanup_errors.append(f"Cleanup failed for {path}: {error}")
    if failure or cleanup_errors:
        raise NemotronInstallError(
            "; ".join(
                filter(
                    None,
                    [
                        f"Nemotron installation failed: {failure}" if failure else None,
                        *cleanup_errors,
                    ],
                )
            )
        ) from failure
    assert result is not None
    return result
