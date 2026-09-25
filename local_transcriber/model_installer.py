"""Explicit fixed-bundle setup, never called by transcription.

Publication requires macOS renameatx_np(RENAME_EXCL) or Linux
renameat2(RENAME_NOREPLACE), and filesystem support. There is no unsafe fallback.
The parent must be user-controlled: this is not a sandbox against a local actor
who can rename parents or tamper with private staging/locks. A crashed install
leaves a lock; it is never automatically stolen or removed by another install.
"""

from __future__ import annotations

import ctypes
import gzip
import hashlib
import json
import os
import shutil
import stat
import sys
import tarfile
import tempfile
import time
from dataclasses import dataclass, replace
from pathlib import Path
from urllib.request import urlopen

from local_transcriber import model_validation as mv

WRAPPER = "gigaam-v3-onnx-int8"
BLOCK_BYTES = 1024 * 1024
MAX_ENTRIES = 12  # wrapper, five files, six possible AppleDouble companions
MAX_METADATA_BYTES = 256 * 1024
MAX_METADATA_FILE_BYTES = 64 * 1024
MAX_PAX_HEADERS = 4
MAX_PAX_HEADER_BYTES = 256
MAX_PAX_BYTES = 768
PAX_FILE_NAMES = frozenset({
    "v3_e2e_rnnt_encoder.onnx", "v3_e2e_rnnt_decoder.onnx",
    "v3_e2e_rnnt_joint.onnx", "v3_e2e_rnnt_tokenizer.model",
})
PAX_KEYS = frozenset({
    b"mtime", b"LIBARCHIVE.xattr.com.apple.provenance",
    b"SCHILY.xattr.com.apple.provenance",
})
DOWNLOAD_TIMEOUT_SECONDS = 30
DOWNLOAD_DEADLINE_SECONDS = 1800


class ModelInstallError(RuntimeError):
    """Installation failed; no overwrite/retry/fallback is implied."""


@dataclass(frozen=True)
class InstallResult:
    model_dir: Path
    metadata: mv.ModelMetadata
    already_installed: bool


def _publish_no_replace(source: Path, target: Path) -> None:
    # POSIX rename can replace an empty directory even after an existence check.
    libc = ctypes.CDLL(None, use_errno=True)
    if sys.platform == "darwin":
        name, flag = "renameatx_np", 0x00000004  # RENAME_EXCL
    elif sys.platform == "linux":
        name, flag = "renameat2", 1  # RENAME_NOREPLACE
    else:
        raise ModelInstallError("Atomic no-replace publication unsupported on this OS")
    try:
        rename = getattr(libc, name)
    except AttributeError:
        raise ModelInstallError(f"Atomic no-replace publication needs {name}") from None
    rename.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    rename.restype = ctypes.c_int
    # Pin the parent for this syscall; source is a child of our sibling workspace.
    with_parent = os.open(target.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        relative_source = source.relative_to(target.parent)
        result = rename(
            with_parent,
            os.fsencode(relative_source),
            with_parent,
            os.fsencode(target.name),
            flag,
        )
        if result != 0:
            code = ctypes.get_errno()
            raise OSError(code, os.strerror(code), str(target))
    finally:
        os.close(with_parent)


def _download(archive: Path, spec: mv.BundleSpec) -> None:
    digest = hashlib.sha256()
    size = 0
    deadline = time.monotonic() + DOWNLOAD_DEADLINE_SECONDS
    with (
        urlopen(spec.archive_url, timeout=DOWNLOAD_TIMEOUT_SECONDS) as response,
        archive.open("xb") as output,
    ):
        while True:
            if time.monotonic() > deadline:
                raise ModelInstallError("Model download exceeded time limit")
            block = response.read(min(BLOCK_BYTES, spec.archive_size_bytes - size + 1))
            if not block:
                break
            size += len(block)
            if size > spec.archive_size_bytes:
                raise ModelInstallError("Model archive exceeds pinned size")
            digest.update(block)
            output.write(block)
    if size != spec.archive_size_bytes:
        raise ModelInstallError("Model archive size does not match pinned size")
    if digest.hexdigest() != spec.archive_sha256:
        raise ModelInstallError("Model archive SHA-256 does not match pinned checksum")


def _check_pax_metadata(payload: bytes) -> None:
    # Values may contain non-UTF-8, NUL and newlines. Only record byte lengths
    # delimit them; never decode/apply timestamps, xattrs or tar layout overrides.
    seen: set[bytes] = set()
    offset = 0
    if not payload:
        raise ModelInstallError("Empty PAX metadata")
    while offset < len(payload):
        space = payload.find(b" ", offset)
        digits = payload[offset:space] if space != -1 else b""
        if (not digits.isdigit() or digits.startswith(b"0")
                or len(digits) > len(str(MAX_PAX_HEADER_BYTES))):
            raise ModelInstallError("Malformed PAX record length")
        end = offset + int(digits)
        if end > len(payload) or end <= space + 1 or payload[end - 1:end] != b"\n":
            raise ModelInstallError("Malformed PAX record framing")
        key, separator, _ = payload[space + 1:end - 1].partition(b"=")
        if not separator or key not in PAX_KEYS:
            raise ModelInstallError("Unsupported PAX metadata key")
        if key in seen:
            raise ModelInstallError("Duplicate PAX metadata key")
        seen.add(key)
        offset = end


def _unpack(archive: Path, raw_tar: Path, staging: Path, spec: mv.BundleSpec) -> None:
    # Bound the *whole* gzip output before parsing tar. This includes headers,
    # padding, AppleDouble, trailing data and unsupported extended metadata.
    limit = (
        sum(f.size_bytes for f in spec.files)
        + MAX_METADATA_BYTES
        + MAX_ENTRIES * 1024
        + MAX_PAX_BYTES
        + MAX_PAX_HEADERS * 1024
        + 10240
    )
    size = 0
    with gzip.open(archive, "rb") as source, raw_tar.open("xb") as output:
        while block := source.read(min(BLOCK_BYTES, limit - size + 1)):
            size += len(block)
            if size > limit:
                raise ModelInstallError("Decompressed archive exceeds byte limit")
            output.write(block)

    expected = {f.name: f for f in spec.files}
    seen: set[str] = set()
    files: dict[str, tuple[int, int]] = {}
    metadata_bytes = 0
    pax_count = 0
    pax_bytes = 0
    pending_pax: str | None = None
    pax_paths = {f"{WRAPPER}/PaxHeader/{name}": f"{WRAPPER}/{name}"
                 for name in PAX_FILE_NAMES if name in expected}
    with raw_tar.open("rb") as source:
        while True:
            header = source.read(512)
            if header == bytes(512):
                if pending_pax is not None:
                    raise ModelInstallError("Dangling PAX metadata")
                if source.read(512) != bytes(512):
                    raise ModelInstallError("Invalid tar end marker")
                while block := source.read(BLOCK_BYTES):
                    if any(block):
                        raise ModelInstallError("Unexpected data after tar end marker")
                break
            if len(header) != 512:
                raise ModelInstallError("Truncated tar header")
            entry = tarfile.TarInfo.frombuf(header, "utf-8", "strict")
            # Parse physical headers: tarfile must not consume extensions for us.
            if pending_pax is not None:
                if (entry.type not in (tarfile.REGTYPE, tarfile.AREGTYPE)
                        or entry.name != pending_pax or entry.linkname):
                    raise ModelInstallError("Unexpected PAX attachment")
                pending_pax = None
            if entry.size < 0:
                raise ModelInstallError("Negative archive entry size")
            end = source.tell() + ((entry.size + 511) // 512) * 512
            if end > size:
                raise ModelInstallError("Truncated tar entry")
            if entry.type == tarfile.XHDTYPE:
                pax_count += 1
                pax_bytes += entry.size
                if pax_count > MAX_PAX_HEADERS:
                    raise ModelInstallError("PAX header count exceeds limit")
                if entry.size > MAX_PAX_HEADER_BYTES or pax_bytes > MAX_PAX_BYTES:
                    raise ModelInstallError("PAX metadata exceeds byte limit")
                if entry.name not in pax_paths or entry.linkname:
                    raise ModelInstallError("Unexpected PAX archive header")
                _check_pax_metadata(source.read(entry.size))
                if any(source.read(end - source.tell())):
                    raise ModelInstallError("Invalid PAX padding")
                pending_pax = pax_paths[entry.name]
                continue
            if entry.type not in (tarfile.REGTYPE, tarfile.AREGTYPE, tarfile.DIRTYPE):
                raise ModelInstallError(
                    "Only regular files and the wrapper are allowed"
                )
            parts = entry.name.split("/")
            if entry.name.startswith("/") or ".." in parts or "\\" in entry.name:
                raise ModelInstallError("Unsafe archive path")
            normalized = "/".join(p for p in parts if p not in ("", "."))
            if normalized in seen:
                raise ModelInstallError("Duplicate archive entry")
            seen.add(normalized)
            if len(seen) > MAX_ENTRIES:
                raise ModelInstallError("Archive exceeds entry limit")
            if entry.isdir():
                if normalized != WRAPPER or entry.size != 0:
                    raise ModelInstallError("Unexpected archive directory")
            else:
                prefix, _, name = normalized.rpartition("/")
                if prefix == WRAPPER and name in expected:
                    if entry.size != expected[name].size_bytes:
                        raise ModelInstallError(f"Incorrect pinned file size: {name}")
                    files[name] = (source.tell(), entry.size)
                elif (
                    prefix == WRAPPER and name.startswith("._") and name[2:] in expected
                ) or normalized == f"._{WRAPPER}":
                    metadata_bytes += entry.size
                    if (
                        entry.size > MAX_METADATA_FILE_BYTES
                        or metadata_bytes > MAX_METADATA_BYTES
                    ):
                        raise ModelInstallError(
                            "AppleDouble metadata exceeds byte limit"
                        )
                else:
                    raise ModelInstallError(f"Unexpected archive entry: {entry.name}")
            source.seek(end)
        if set(files) != set(expected):
            raise ModelInstallError("Archive is missing required model files")
        # All headers/layout/limits are checked before creating any model file.
        for name, (offset, count) in files.items():
            source.seek(offset)
            with (staging / name).open("xb") as output:
                while count:
                    block = source.read(min(BLOCK_BYTES, count))
                    if not block:
                        raise ModelInstallError("Truncated model file")
                    output.write(block)
                    count -= len(block)


def _identity(path: Path) -> tuple[int, int]:
    info = path.lstat()
    return info.st_dev, info.st_ino


def _validate(path: Path) -> mv.ModelMetadata:
    with mv.load_model_directory(path) as loaded:
        metadata = loaded.metadata
        if metadata.verification != "pinned":
            raise ModelInstallError("Model is not the exact pinned bundle")
    return metadata


def install_model(target: Path | None = None) -> InstallResult:
    """Install into explicit target or managed cache, ignoring runtime env/legacy.

    Returns metadata only; temporary runtime ownership ends before writing the
    manifest or publishing. Existing exact pinned targets are read-only successes.
    Any other existing target (including an empty directory/symlink) is preserved.
    Errors, including cleanup errors after publication, raise ModelInstallError;
    after such an error inspect the target before retrying. No force mode exists.
    """
    workspace = None
    workspace_identity = None
    lock_identity = None
    lock = None
    result = None
    failure = None
    cleanup_errors = []
    try:
        requested = (
            target if target is not None else mv.managed_cache_dir()
        ).expanduser()
        # Resolve parents, not the final component: do not follow a target symlink.
        target = requested.parent.resolve() / requested.name
        target.parent.mkdir(parents=True, exist_ok=True)
        lock = target.with_name(f".{target.name}.install-lock")
        try:
            lock.mkdir(mode=0o700)
        except FileExistsError:
            raise ModelInstallError(
                f"Model target is locked: {lock}; no automatic stale-lock removal"
            ) from None
        lock_identity = _identity(lock)
        try:
            target_info = target.lstat()
        except FileNotFoundError:
            target_info = None
        if target_info is not None:
            try:
                if not stat.S_ISDIR(target_info.st_mode):
                    raise ModelInstallError("Target is not a regular directory")
                metadata = _validate(target)
                if _identity(target) != (target_info.st_dev, target_info.st_ino):
                    raise ModelInstallError("Target changed during validation")
            except (mv.ModelValidationError, ModelInstallError) as error:
                raise ModelInstallError(
                    f"Existing target preserved; choose an absent target path: "
                    f"{target}: {error}"
                ) from None
            result = InstallResult(target, metadata, True)
        else:
            spec = mv.BUNDLE
            workspace = Path(
                tempfile.mkdtemp(prefix=f".{target.name}.install-", dir=target.parent)
            )
            workspace_identity = _identity(workspace)
            staging = workspace / "model"
            staging.mkdir()
            archive = workspace / "archive.tar.gz"
            _download(archive, spec)
            _unpack(archive, workspace / "archive.tar", staging, spec)
            metadata = _validate(staging)
            manifest = metadata.install_manifest()
            with (staging / mv.MANIFEST_NAME).open("x", encoding="utf-8") as output:
                json.dump(manifest, output, ensure_ascii=False, indent=2)
                output.write("\n")
            _publish_no_replace(staging, target)
            result = InstallResult(
                target,
                replace(metadata, model_dir=target, manifest_integrity=True),
                False,
            )
    except Exception as error:
        failure = f"GigaAM installation failed: {error}"
    finally:
        for path, identity, recursive in (
            (workspace, workspace_identity, True),
            (lock, lock_identity, False),
        ):
            if identity is not None and path is not None:
                try:
                    if _identity(path) != identity:
                        raise ModelInstallError(
                            f"Owned path replaced; left untouched: {path}"
                        )
                    if recursive:
                        shutil.rmtree(path)
                    else:
                        path.rmdir()
                except Exception as error:
                    cleanup_errors.append(f"Cleanup failed for {path}: {error}")
    if failure or cleanup_errors:
        raise ModelInstallError("; ".join(filter(None, [failure, *cleanup_errors])))
    assert result is not None
    return result
