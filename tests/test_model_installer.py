"""Synthetic installer safety tests; no real model, transport or inference."""

from __future__ import annotations

import gc
import gzip
import hashlib
import io
import json
import os
import shutil
import tarfile
import tempfile
import threading
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import yaml

from local_transcriber.models import installer as mi
from local_transcriber.models import validation as mv
from tests.test_model_validation import CONFIG, FakeRuntime

WRAPPER = "gigaam-v3-onnx-int8"


def member(name, data=b"", kind=tarfile.REGTYPE):
    entry = tarfile.TarInfo(name)
    entry.type = kind
    entry.size = len(data)
    if kind in (tarfile.SYMTYPE, tarfile.LNKTYPE):
        entry.linkname = "outside"
    return entry, data


def archive_bytes(entries):
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w", format=tarfile.USTAR_FORMAT) as archive:
        for entry, data in entries:
            archive.addfile(entry, io.BytesIO(data))
    return gzip.compress(output.getvalue(), mtime=0)


def pax_record(key, value):
    body = key + b"=" + value + b"\n"
    size = len(body) + 2
    while True:
        record = str(size).encode() + b" " + body
        if len(record) == size:
            return record
        size = len(record)


def pax_member(name, payload=None):
    if payload is None:
        # Invented values only; embedded newlines/NUL/non-UTF-8 are opaque bytes.
        payload = (pax_record(b"mtime", b"1000000000.123456789")
                   + pax_record(b"LIBARCHIVE.xattr.com.apple.provenance",
                                b"synthetic-value")
                   + pax_record(b"SCHILY.xattr.com.apple.provenance",
                                b"\x01\x02\x00\xff\n=\x80fake"))
    return member(f"{WRAPPER}/PaxHeader/{name}", payload, tarfile.XHDTYPE)


class FakeResponse(io.BytesIO):
    def __init__(self, data, reads, failure=None):
        super().__init__(data)
        self.reads = reads
        self.failure = failure

    def read(self, size=-1):
        if not 0 < size <= 1024 * 1024:
            raise AssertionError(f"Unbounded download read: {size}")
        self.reads.append(size)
        if self.failure and self.tell():
            raise self.failure
        # Force many partial reads; one read is not necessarily the whole response.
        return super().read(min(size, 97))


class ModelInstallerTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name).resolve()
        self.target = self.root / "installed"
        self.contents = {
            f.name: f"synthetic {f.name}".encode() for f in mv.BUNDLE.files
        }
        self.contents["v3_e2e_rnnt.yaml"] = yaml.safe_dump(CONFIG).encode()
        self.entries = [member(WRAPPER + "/", kind=tarfile.DIRTYPE)] + [
            member(f"{WRAPPER}/{name}", data) for name, data in self.contents.items()
        ]
        self.data = archive_bytes(self.entries)
        self.spec = replace(
            mv.BUNDLE,
            archive_url="https://invalid.example/synthetic-test-only.tar.gz",
            archive_size_bytes=len(self.data),
            archive_sha256=hashlib.sha256(self.data).hexdigest(),
            files=tuple(mv.FileSpec(name, len(data), hashlib.sha256(data).hexdigest())
                        for name, data in self.contents.items()),
        )
        # Replace the immutable trust root only within this test, shared by loader
        # and installer. Never mutate/relax production hashes, sizes or limits.
        self.enterContext(patch.object(mv, "BUNDLE", self.spec))
        self.runtime = self.enterContext(FakeRuntime().installed())
        self.reads = []
        self.responses = []
        self.transport_failure = None
        self.transport = self.enterContext(
            patch.object(mi, "urlopen", self.open_response)
        )
        self.network_calls = []

    def open_response(self, url, *, timeout):
        self.network_calls.append((url, timeout))
        response = FakeResponse(self.data, self.reads, self.transport_failure)
        self.responses.append(response)
        return response

    def set_archive(self, entries=None, *, raw=None, files=None):
        self.data = archive_bytes(entries) if raw is None else raw
        self.spec = replace(self.spec, archive_size_bytes=len(self.data),
                            archive_sha256=hashlib.sha256(self.data).hexdigest(),
                            files=self.spec.files if files is None else files)
        self.enterContext(patch.object(mv, "BUNDLE", self.spec))

    def assert_released(self):
        gc.collect()
        self.assertTrue(all(ref() is None for ref in self.runtime.refs))
        self.assertFalse(any(self.runtime.runs.values()))
        self.assertEqual(self.runtime.decoded, [])
        self.assertTrue(all(response.closed for response in self.responses))

    def assert_clean_failure(self, message):
        # Retain the exception while checking weakrefs: traceback ownership matters.
        retained = None
        real_cleanup = mi.shutil.rmtree

        def cleanup(path):
            self.assert_released()
            real_cleanup(path)

        try:
            with patch.object(mi.shutil, "rmtree", cleanup):
                mi.install_model(self.target)
        except mi.ModelInstallError as error:
            retained = error
        self.assertIsNotNone(retained)
        self.assertRegex(str(retained), message)
        self.assert_released()
        self.assertEqual(list(self.root.iterdir()), [])
        return retained

    def write_existing(self):
        self.target.mkdir()
        for name, data in self.contents.items():
            (self.target / name).write_bytes(data)

    def snapshot(self):
        return {p.name: (p.read_bytes(), p.stat().st_mtime_ns)
                for p in self.target.iterdir()}

    def observed_pax_entries(self):
        # Raw preflight order, including YAML before the encoder triple.
        entries = [member(WRAPPER + "/", kind=tarfile.DIRTYPE)]
        for suffix in ("joint.onnx", "decoder.onnx", "tokenizer.model",
                       "yaml", "encoder.onnx"):
            name = "v3_e2e_rnnt" + ("." if suffix == "yaml" else "_") + suffix
            if suffix != "yaml":
                extension = pax_member(name)
                if suffix == "tokenizer.model":
                    extension = pax_member(
                        name, extension[1].replace(b"30 mtime=1000000000.123456789\n",
                                                   b"29 mtime=1000000000.12345678\n"))
                entries.extend([member(f"{WRAPPER}/._{name}", b"a" * 163),
                                extension])
            entries.append(member(f"{WRAPPER}/{name}", self.contents[name]))
        return entries

    def test_pinned_archive_metadata_shape_publication_and_idempotency(self):
        entries = self.observed_pax_entries()
        self.assertEqual(len(entries), 14)
        self.assertEqual([entry.type for entry, _ in entries],
                         [b"5", b"0", b"x", b"0", b"0", b"x", b"0",
                          b"0", b"x", b"0", b"0", b"0", b"x", b"0"])
        self.assertEqual([len(data) for entry, data in entries
                          if entry.type == tarfile.XHDTYPE], [136, 136, 135, 136])
        self.set_archive(entries)
        real_publish = mi._publish_no_replace

        def publish(source, target):
            self.assert_released()
            self.assertFalse(target.exists())
            self.assertEqual({p.name for p in source.iterdir()},
                             set(self.contents) | {mv.MANIFEST_NAME})
            real_publish(source, target)

        with patch.object(mi, "_publish_no_replace", publish):
            result = mi.install_model(self.target)
        self.assertFalse(result.already_installed)
        self.assertEqual(result.metadata.verification, "pinned")
        self.assertTrue(result.metadata.manifest_integrity)
        for name, data in self.contents.items():
            self.assertEqual((self.target / name).read_bytes(), data)
        before = self.snapshot()
        with patch.object(mi, "urlopen", side_effect=AssertionError("offline")):
            self.assertTrue(mi.install_model(self.target).already_installed)
        self.assertEqual(self.snapshot(), before)
        self.assertEqual(list(self.root.iterdir()), [self.target])
        self.assert_released()

    def assert_pax_failure(self, entries, message):
        self.set_archive(entries)
        with (patch.object(mi, "_validate") as validate,
              patch.object(mi, "_publish_no_replace") as publish):
            self.assert_clean_failure(message)
        validate.assert_not_called()
        publish.assert_not_called()
        self.assertEqual(self.runtime.session_calls, [])

    def test_pax_rejects_layout_unknown_sparse_and_duplicate_keys(self):
        name = "v3_e2e_rnnt_joint.onnx"
        for key in (b"path", b"linkpath", b"size", b"unknown", b"GNU.sparse.map",
                    b"SCHILY.realsize", b"\xff", b"", b"mtime\x00"):
            with self.subTest(key=key):
                entries = self.observed_pax_entries()
                entries[2] = pax_member(name, pax_record(key, b"outside"))
                self.assert_pax_failure(entries, "Unsupported PAX")
        for key in (b"mtime", b"LIBARCHIVE.xattr.com.apple.provenance",
                    b"SCHILY.xattr.com.apple.provenance"):
            with self.subTest(duplicate=key):
                entries = self.observed_pax_entries()
                entries[2] = pax_member(name, pax_record(key, b"a") * 2)
                self.assert_pax_failure(entries, "Duplicate PAX")

    def test_pax_rejects_malformed_byte_framing(self):
        good = pax_record(b"mtime", b"1")
        malformed = (b"", b"mtime=1\n", b"0 mtime=1\n", b"-1 mtime=1\n",
                     b"+11 mtime=1\n", b"011 mtime=1\n", b"9999 mtime=1\n",
                     b"1 mtime=1\n", b"99 mtime=1\n", b"11 mtime=1!",
                     b"11 mtime:1\n", good[:-1], good + b"junk",
                     b"10 mtime=\xff\n")
        for payload in malformed:
            with self.subTest(payload=payload):
                entries = self.observed_pax_entries()
                entries[2] = pax_member("v3_e2e_rnnt_joint.onnx", payload)
                self.assert_pax_failure(entries, "PAX")

    def test_pax_rejects_global_gnu_and_unexpected_attachments(self):
        entries = self.observed_pax_entries()
        extension = entries[2]
        for kind in (tarfile.XGLTYPE, tarfile.GNUTYPE_LONGNAME,
                     tarfile.GNUTYPE_LONGLINK, tarfile.GNUTYPE_SPARSE):
            with self.subTest(kind=kind):
                changed = entries.copy()
                changed[2] = member(extension[0].name, extension[1], kind)
                self.assert_pax_failure(changed, "regular")
        cases = {
            "dangling": self.entries + [extension],
            "consecutive": entries[:3] + entries[2:],
            "wrong file": entries[:3] + [entries[6]] + entries[3:],
            "directory": entries[:3] + [entries[0]] + entries[3:],
            "appledouble": entries[:3] + [entries[1]] + entries[3:],
            "alias": entries[:3] + [member(
                f"{WRAPPER}/./v3_e2e_rnnt_joint.onnx", entries[3][1])]
                + entries[4:],
            "link": entries[:3] + [member(entries[3][0].name,
                                         kind=tarfile.SYMTYPE)] + entries[4:],
            "yaml": self.entries + [pax_member("v3_e2e_rnnt.yaml")],
            "unknown path": self.entries + [pax_member("../outside")],
            "duplicate file": entries[:4] + entries[2:],
        }
        for case, changed in cases.items():
            with self.subTest(case=case):
                self.assert_pax_failure(changed, "PAX|Duplicate archive")
        for index in (2, 3):
            changed = self.observed_pax_entries()
            changed[index][0].linkname = "outside"
            self.assert_pax_failure(changed, "PAX")

    def test_pax_count_payload_and_logical_entry_limits(self):
        entries = self.observed_pax_entries()
        self.assert_pax_failure(entries + [entries[2]], "PAX header count")
        changed = entries.copy()
        changed[2] = pax_member("v3_e2e_rnnt_joint.onnx",
                                pax_record(b"mtime", b"x" * 256))
        self.assert_pax_failure(changed, "PAX metadata exceeds")
        # Four individually legal payloads exceed the independent aggregate cap.
        changed = [(pax_member(e.name.rsplit("/", 1)[-1],
                               pax_record(b"mtime", b"x" * 190))
                    if e.type == tarfile.XHDTYPE else (e, d)) for e, d in entries]
        self.assert_pax_failure(changed, "PAX metadata exceeds")
        self.assert_pax_failure(entries + [
            member(f"{WRAPPER}/._v3_e2e_rnnt.yaml"), member("._" + WRAPPER),
            member("extra")], "entry limit")
        raw = gzip.decompress(archive_bytes(entries))
        self.set_archive(raw=gzip.compress(raw + bytes(400000)))
        self.assert_clean_failure("Decompressed archive exceeds")

    def test_pax_truncated_payload_padding_and_headers_create_no_files(self):
        entries = self.observed_pax_entries()
        raw = gzip.decompress(archive_bytes(entries))
        for changed, message in (
            (raw[:512 * 4 + 20], "Truncated tar entry"),
            (raw[:512 * 5 + 20], "Truncated tar header"),
            (raw[:512 * 4 + 136] + b"!" + raw[512 * 4 + 137:], "PAX padding"),
        ):
            with self.subTest(message=message):
                archive = self.root / "input.gz"
                raw_tar = self.root / "raw.tar"
                staging = self.root / "staging"
                archive.write_bytes(gzip.compress(changed))
                staging.mkdir()
                try:
                    with self.assertRaisesRegex(mi.ModelInstallError, message):
                        mi._unpack(archive, raw_tar, staging, self.spec)
                    self.assertEqual(list(staging.iterdir()), [])
                finally:
                    archive.unlink()
                    raw_tar.unlink()
                    staging.rmdir()

    def test_happy_path_shared_loader_manifest_release_before_publication(self):
        self.set_archive(self.entries + [member(f"{WRAPPER}/._{name}", b"metadata")
                                        for name in self.contents])
        real_publish = mi._publish_no_replace

        def publish(source, target):
            self.assert_released()
            self.assertEqual(source.parent.parent, target.parent)
            self.assertEqual(source.stat().st_dev, target.parent.stat().st_dev)
            self.assertFalse(target.exists())
            manifest = json.loads((source / mv.MANIFEST_NAME).read_text())
            self.assertEqual(manifest["schema_version"], 1)
            self.assertEqual(manifest["profile"], "v3_e2e_rnnt")
            self.assertEqual(manifest["bundle_sha256"], self.spec.archive_sha256)
            self.assertEqual(manifest["versions"]["onnxruntime"], "fake-ort")
            self.assertEqual(manifest["files"], {
                name: {
                    "size_bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()
                }
                for name, data in self.contents.items()
            })
            real_publish(source, target)

        with (patch.object(mi, "_publish_no_replace", publish),
              patch.object(
                  mv, "load_model_directory", wraps=mv.load_model_directory
              ) as load,
              patch.object(mv, "hash_file", wraps=mv.hash_file) as hashes):
            result = mi.install_model(self.target)
        self.assertFalse(result.already_installed)
        self.assertEqual(result.model_dir, self.target)
        self.assertEqual(result.metadata.model_dir, self.target)
        self.assertTrue(result.metadata.manifest_integrity)
        self.assertEqual(result.metadata.verification, "pinned")
        self.assertEqual(load.call_count, 1)
        self.assertEqual(hashes.call_count, 5)
        self.assertEqual(len(self.runtime.session_calls), 3)
        self.assertEqual(len(self.runtime.tokenizer_paths), 1)
        self.assertEqual(self.network_calls, [(self.spec.archive_url, 30)])
        self.assertGreater(len(self.reads), 1)
        self.assertEqual(list(self.root.iterdir()), [self.target])
        self.assertEqual(set(p.name for p in self.target.iterdir()),
                         set(self.contents) | {"model-manifest.json"})
        for name, data in self.contents.items():
            self.assertEqual((self.target / name).read_bytes(), data)

    def test_existing_pinned_is_unchanged_and_offline_with_or_without_manifest(self):
        self.write_existing()
        for has_manifest in (False, True):
            if has_manifest:
                with mv.load_model_directory(self.target) as loaded:
                    (self.target / mv.MANIFEST_NAME).write_text(
                        json.dumps(loaded.metadata.install_manifest()))
            before = self.snapshot()
            with patch.object(mv, "hash_file", wraps=mv.hash_file) as hashes:
                result = mi.install_model(self.target)
            self.assertTrue(result.already_installed)
            self.assertEqual(result.metadata.manifest_integrity, has_manifest)
            self.assertEqual(hashes.call_count, 5)
            self.assertEqual(self.snapshot(), before)
            self.assertEqual(self.network_calls, [])
            self.assert_released()
        self.assertEqual(list(self.root.iterdir()), [self.target])

    def test_existing_other_self_consistent_manifest_cannot_claim_pinned(self):
        self.write_existing()
        path = self.target / "v3_e2e_rnnt_encoder.onnx"
        path.write_bytes(b"other weights")
        manifest = {
            "schema_version": 1, "profile": "v3_e2e_rnnt",
            "bundle_sha256": self.spec.archive_sha256,
            "versions": {"fake": "1"},
            "files": {p.name: {"size_bytes": len(p.read_bytes()),
                              "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                      for p in self.target.iterdir()},
        }
        (self.target / mv.MANIFEST_NAME).write_text(json.dumps(manifest))
        before = self.snapshot()
        with self.assertRaisesRegex(mi.ModelInstallError, "choose an absent target"):
            mi.install_model(self.target)
        self.assertEqual(self.snapshot(), before)
        self.assertEqual(self.network_calls, [])
        self.assert_released()

    def test_existing_empty_invalid_corrupt_manifest_file_and_symlink_are_preserved(
        self,
    ):
        for case in ("empty", "invalid", "manifest", "file", "symlink"):
            with self.subTest(case=case):
                if case == "file":
                    self.target.write_bytes(b"keep")
                elif case == "symlink":
                    self.target.symlink_to(self.root / "missing")
                elif case == "manifest":
                    self.write_existing()
                    (self.target / mv.MANIFEST_NAME).write_bytes(b"{bad json")
                else:
                    self.target.mkdir()
                    if case == "invalid":
                        (self.target / "keep").write_bytes(b"old model")
                before = self.snapshot() if self.target.is_dir() else None
                inode = self.target.lstat().st_ino
                with self.assertRaisesRegex(
                    mi.ModelInstallError, "Existing target preserved"
                ):
                    mi.install_model(self.target)
                self.assertEqual(self.target.lstat().st_ino, inode)
                if before is not None:
                    self.assertEqual(self.snapshot(), before)
                    shutil.rmtree(self.target)
                else:
                    self.target.unlink()
                self.assertEqual(self.network_calls, [])
                self.assertEqual(list(self.root.iterdir()), [])

    def test_bad_checksum_and_size_are_rejected_before_decompression_or_load(self):
        original = self.data
        for data, message in ((original[:-1], "size"), (original + b"x", "size"),
                              (bytes([original[0] ^ 1]) + original[1:], "SHA-256")):
            with self.subTest(message=message):
                self.data = data
                with patch.object(mi, "_unpack") as unpack:
                    self.assert_clean_failure(message)
                unpack.assert_not_called()
        self.assertEqual(self.runtime.session_calls, [])

    def test_timeout_interruption_and_deadline_cleanup(self):
        for error in (TimeoutError("fake timeout"), OSError("connection interrupted")):
            self.transport_failure = error
            self.assert_clean_failure(str(error))
        self.transport_failure = None
        with patch.object(mi.time, "monotonic", side_effect=[0, 1801]):
            self.assert_clean_failure("time limit")
        with patch.object(mi, "urlopen", side_effect=OSError("connect failed")):
            self.assert_clean_failure("connect failed")

    def test_archive_path_type_layout_and_duplicate_rejections(self):
        name = next(iter(self.contents))
        bad = [
            member("/absolute"), member(f"{WRAPPER}/../outside"),
            member(f"{WRAPPER}/sub/../../outside"), member("C:\\outside"),
            member(f"wrong/{name}"), member(name), member(f"{WRAPPER}/extra"),
            member(f"{WRAPPER}/sub/", kind=tarfile.DIRTYPE),
            member(f"{WRAPPER}/{name}", kind=tarfile.SYMTYPE),
            member(f"{WRAPPER}/._{name}", kind=tarfile.LNKTYPE),
            member(f"{WRAPPER}/._{name}", kind=tarfile.FIFOTYPE),
            member(f"{WRAPPER}/._{name}", kind=tarfile.CHRTYPE),
            member(f"{WRAPPER}/._{name}", kind=tarfile.BLKTYPE),
            member(f"{WRAPPER}/._{name}", kind=tarfile.GNUTYPE_SPARSE),
            member(f"{WRAPPER}/../._{name}"),
            member(f"{WRAPPER}/./{name}"), member(f"{WRAPPER}//{name}"),
            member(f"./{WRAPPER}/{name}"), member(f"{WRAPPER}/" + name),
            member("pax", b"path=outside", tarfile.XHDTYPE),
            member("longname", b"outside", tarfile.GNUTYPE_LONGNAME),
        ]
        for entry in bad:
            with self.subTest(name=entry[0].name, kind=entry[0].type):
                self.set_archive(self.entries + [entry])
                self.assert_clean_failure("archive|regular|Duplicate")
        self.assertEqual(self.runtime.session_calls, [])

    def test_missing_file_wrong_file_size_and_invalid_gzip_tar(self):
        self.set_archive(self.entries[:-1])
        self.assert_clean_failure("missing")
        first = self.entries[1][0].name
        self.set_archive(
            [self.entries[0], member(first, b"oversized")] + self.entries[2:]
        )
        self.assert_clean_failure("file size")
        for raw in (b"not gzip", gzip.compress(b"not tar"),
                    gzip.compress(b"x" * 1024)):
            self.set_archive(raw=raw)
            self.assert_clean_failure("failed")

    def test_entry_metadata_and_total_decompressed_limits(self):
        metadata = [member(f"{WRAPPER}/._{name}", b"x") for name in self.contents]
        allowed = self.entries + metadata + [member("._" + WRAPPER, b"x")]
        self.set_archive(allowed + [member("extra")])
        self.assert_clean_failure("entry limit")
        self.set_archive(self.entries + [member(
            f"{WRAPPER}/._{next(iter(self.contents))}", b"x" * (64 * 1024 + 1)
        )])
        self.assert_clean_failure("metadata exceeds")
        self.set_archive(self.entries + [member(f"{WRAPPER}/._{name}", b"x" * 53000)
                                         for name in self.contents])
        self.assert_clean_failure("metadata exceeds")
        # Padding and trailing bytes count even if tar iteration would stop early.
        raw_tar = gzip.decompress(archive_bytes(self.entries))
        self.set_archive(raw=gzip.compress(raw_tar + bytes(400000)))
        self.assert_clean_failure("Decompressed archive exceeds")
        self.set_archive(raw=gzip.compress(raw_tar + b"hidden"))
        self.assert_clean_failure("after tar end")

    def test_real_loader_rejects_yaml_signatures_partial_loads_and_unpinned_bytes(self):
        yaml_name = "v3_e2e_rnnt.yaml"
        invalid = b"preprocessor: {}\n"
        files = tuple(replace(f, size_bytes=len(invalid),
                              sha256=hashlib.sha256(invalid).hexdigest())
                      if f.name == yaml_name else f for f in self.spec.files)
        entries = [(e, d) if not e.name.endswith(yaml_name)
                   else member(e.name, invalid) for e, d in self.entries]
        original_spec = self.spec
        self.set_archive(entries, files=files)
        self.assert_clean_failure("preprocessor")
        self.set_archive(self.entries, files=original_spec.files)
        for failure in ("decoder", "joint", "tokenizer_load"):
            self.runtime.fail = failure
            self.assert_clean_failure("synthetic")
        self.runtime.fail = None
        self.runtime.signatures["encoder"][1][1] = (
            "encoded_len", "tensor(int64)", ["B"])
        self.assert_clean_failure("dtype/rank")
        self.runtime.signatures["encoder"][1][1] = (
            "encoded_len", "tensor(int32)", ["B"])
        changed = self.contents["v3_e2e_rnnt_encoder.onnx"].replace(
            b"synthetic", b"different"
        )
        entries = [(e, d) if not e.name.endswith("encoder.onnx")
                   else member(e.name, changed) for e, d in self.entries]
        self.set_archive(entries)
        self.assert_clean_failure("not the exact pinned")

    def assert_interrupted_load_cleanup(self, interruption):
        real_check = mv._check_tensors
        real_cleanup = mi.shutil.rmtree
        refs_start = len(self.runtime.refs)
        before_cleanup = []
        retained = None

        def interrupted_check(nodes, expected, label):
            real_check(nodes, expected, label)
            if label == "decoder inputs":
                raise interruption

        def cleanup(path):
            # Snapshot before actual deletion, while the original traceback is live.
            # Do not abort rmtree on RED: also prove real staging/lock cleanup ran.
            gc.collect()
            before_cleanup.append((
                interruption.__traceback__ is not None,
                [ref() is not None for ref in self.runtime.refs[refs_start:]],
                (path / "model" / "v3_e2e_rnnt_decoder.onnx").is_file(),
            ))
            real_cleanup(path)

        with (
            patch.object(mv, "_check_tensors", interrupted_check),
            patch.object(mi.shutil, "rmtree", cleanup),
        ):
            try:
                mi.install_model(self.target)
            except BaseException as error:
                retained = error
        self.assertIs(retained, interruption)
        self.assertIsNotNone(interruption.__traceback__)
        if isinstance(interruption, SystemExit):
            self.assertEqual(interruption.code, 23)
        self.assertFalse(self.target.exists())
        self.assertEqual(list(self.root.iterdir()), [])
        self.assertEqual(len(self.runtime.refs) - refs_start, 2)
        self.assertEqual(before_cleanup, [(True, [False, False], True)])
        self.assert_released()

    def test_keyboard_interrupt_releases_runtime_before_staging_cleanup(self):
        self.assert_interrupted_load_cleanup(KeyboardInterrupt("synthetic interrupt"))

    def test_system_exit_releases_runtime_before_staging_cleanup(self):
        self.assert_interrupted_load_cleanup(SystemExit(23))

    def test_download_extract_manifest_write_and_rename_errors_cleanup(self):
        real_open = Path.open
        for filename in ("archive.tar.gz", "archive.tar", "v3_e2e_rnnt_encoder.onnx",
                         mv.MANIFEST_NAME):
            def denied(path, mode="r", *args, filename=filename, **kwargs):
                if path.name == filename and "x" in mode:
                    raise OSError("synthetic disk full")
                return real_open(path, mode, *args, **kwargs)
            with self.subTest(filename=filename), patch.object(Path, "open", denied):
                self.assert_clean_failure("disk full")
        with patch.object(
            mi, "_publish_no_replace", side_effect=OSError("rename denied")
        ):
            self.assert_clean_failure("rename denied")

    def test_native_publish_never_replaces_racing_target_even_empty_directory(self):
        real_publish = mi._publish_no_replace
        for kind in ("empty", "model", "file", "symlink"):
            before = {}
            inode = []
            def race(source, target, kind=kind, before=before, inode=inode):
                self.assert_released()
                if kind == "file":
                    target.write_bytes(b"foreign file")
                elif kind == "symlink":
                    target.symlink_to(self.root / "absent")
                else:
                    target.mkdir()
                    if kind == "model":
                        for name, data in self.contents.items():
                            (target / name).write_bytes(data)
                        before.update(self.snapshot())
                inode.append(target.lstat().st_ino)
                real_publish(source, target)
            with (
                self.subTest(kind=kind),
                patch.object(mi, "_publish_no_replace", race),
                self.assertRaises(mi.ModelInstallError),
            ):
                mi.install_model(self.target)
            self.assertEqual(self.target.lstat().st_ino, inode[0])
            self.assertEqual(list(self.root.iterdir()), [self.target])
            if kind in ("empty", "model"):
                self.assertEqual(self.snapshot(), before)
                shutil.rmtree(self.target)
            else:
                if kind == "file":
                    self.assertEqual(self.target.read_bytes(), b"foreign file")
                self.target.unlink()
            self.assert_released()

    def test_unsupported_platform_and_filesystem_fail_closed(self):
        with patch.object(mi.sys, "platform", "unsupported"):
            self.assert_clean_failure("unsupported on this OS")
        # libc failure must not trigger a plain rename fallback.
        with patch.object(
            mi.ctypes, "CDLL", side_effect=OSError("unsupported filesystem")
        ):
            self.assert_clean_failure("unsupported filesystem")

    def test_target_replacement_during_existing_validation_is_detected(self):
        self.write_existing()
        real_loader = mv.load_model_directory
        moved = self.root / "original"
        def replace_target(path):
            loaded = real_loader(path)
            path.rename(moved)
            path.mkdir()
            return loaded
        with (
            patch.object(mv, "load_model_directory", replace_target),
            self.assertRaisesRegex(mi.ModelInstallError, "Target changed"),
        ):
            mi.install_model(self.target)
        self.assertEqual(list(self.target.iterdir()), [])
        for name, data in self.contents.items():
            self.assertEqual((moved / name).read_bytes(), data)
        self.assertEqual(self.network_calls, [])
        self.assert_released()

    def test_concurrent_installation_fails_before_network_and_leaves_owner_lock(self):
        entered = threading.Event()
        release = threading.Event()
        results = []
        errors = []
        real_transport = self.open_response
        def paused_transport(url, *, timeout):
            entered.set()
            if not release.wait(10):
                raise AssertionError("test synchronization timed out")
            return real_transport(url, timeout=timeout)
        def worker():
            try:
                results.append(mi.install_model(self.target))
            except Exception as error:
                errors.append(error)
        with patch.object(mi, "urlopen", paused_transport):
            thread = threading.Thread(target=worker)
            thread.start()
            try:
                self.assertTrue(entered.wait(10))
                lock = self.root / ".installed.install-lock"
                inode = lock.stat().st_ino
                with self.assertRaisesRegex(mi.ModelInstallError, "locked"):
                    mi.install_model(self.target)
                self.assertEqual(lock.stat().st_ino, inode)
                self.assertEqual(self.network_calls, [])
            finally:
                release.set()
                thread.join(10)
            self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(len(results), 1)
        self.assertEqual(len(self.network_calls), 1)
        self.assertEqual(list(self.root.iterdir()), [self.target])
        self.assert_released()

    def test_foreign_lock_and_unrelated_staging_are_never_cleaned(self):
        lock = self.root / ".installed.install-lock"
        lock.mkdir()
        (lock / "owner").write_bytes(b"foreign")
        unrelated = self.root / ".installed.install-foreign"
        unrelated.mkdir()
        (unrelated / "keep").write_bytes(b"keep")
        with self.assertRaisesRegex(mi.ModelInstallError, "locked"):
            mi.install_model(self.target)
        self.assertEqual((lock / "owner").read_bytes(), b"foreign")
        self.assertEqual((unrelated / "keep").read_bytes(), b"keep")
        self.assertEqual(self.network_calls, [])
        shutil.rmtree(lock)
        self.data = b"bad"
        with self.assertRaises(mi.ModelInstallError):
            mi.install_model(self.target)
        self.assertEqual(list(self.root.iterdir()), [unrelated])

    def test_cleanup_errors_are_reported_after_publication_and_lock_still_released(
        self,
    ):
        with (
            patch.object(mi.shutil, "rmtree", side_effect=OSError("cleanup denied")),
            self.assertRaisesRegex(mi.ModelInstallError, "Cleanup failed"),
        ):
            mi.install_model(self.target)
        self.assertTrue((self.target / mv.MANIFEST_NAME).is_file())
        self.assertFalse((self.root / ".installed.install-lock").exists())
        self.assert_released()
        # Do not pretend publication failed atomically: retry discovers pinned target.
        self.assertTrue(mi.install_model(self.target).already_installed)
        self.assertEqual(len(self.network_calls), 1)

    def test_replaced_lock_or_workspace_is_left_untouched(self):
        for resource in ("lock", "workspace"):
            saved = self.root / "saved"
            replaced = []
            def replace_owned(url, *, timeout, resource=resource, saved=saved,
                                          replaced=replaced):
                owned = (self.root / ".installed.install-lock" if resource == "lock"
                         else next(p for p in self.root.iterdir()
                                   if p.name.startswith(".installed.install-")
                                   and p.name != ".installed.install-lock"))
                owned.rename(saved)
                owned.mkdir()
                (owned / "foreign").write_bytes(b"keep")
                replaced.append(owned)
                raise OSError("transport stopped")
            with (
                patch.object(mi, "urlopen", replace_owned),
                self.assertRaisesRegex(mi.ModelInstallError, "Owned path replaced"),
            ):
                mi.install_model(self.target)
            self.assertEqual((replaced[0] / "foreign").read_bytes(), b"keep")
            shutil.rmtree(replaced[0])
            shutil.rmtree(saved)
            self.assertEqual(list(self.root.iterdir()), [])

    def test_default_cache_explicit_path_and_no_runtime_resolver_fallback(self):
        with patch.dict(os.environ, {"XDG_CACHE_HOME": str(self.root / "cache"),
                                     "GIGAAM_MODEL_DIR": str(self.root / "unused")}):
            result = mi.install_model()
        expected = self.root / "cache/local-transcriber/models" / self.spec.cache_name
        self.assertEqual(result.model_dir, expected)
        self.assertFalse((self.root / "unused").exists())
        with patch.dict(os.environ, {"HOME": str(self.root)}):
            explicit = mi.install_model(Path("~/explicit"))
        self.assertEqual(explicit.model_dir, self.root / "explicit")


if __name__ == "__main__":
    unittest.main()
