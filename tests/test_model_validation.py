"""Offline contracts from PLAN/CONTEXT, not evidence of real bundle/ASR parity."""

from __future__ import annotations

import gc
import hashlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
import weakref
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from copy import deepcopy
from dataclasses import replace
from importlib.abc import Loader, MetaPathFinder
from importlib.util import spec_from_loader
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import numpy as np
import yaml

from local_transcriber import cli
from local_transcriber import model_validation as mv
from local_transcriber.gigaam import GigaAMEngine
from local_transcriber.transcript import TranscriptResult

# Independently spelled-out YAML/signatures from the fixed profile contract.
CONFIG: dict[str, Any] = {
    "preprocessor": {
        "sample_rate": 16000,
        "features": 64,
        "n_fft": 320,
        "win_length": 320,
        "hop_length": 160,
        "center": False,
        "mel_scale": "htk",
        "mel_norm": None,
    },
    "head": {
        "decoder": {
            "pred_hidden": 320,
            "pred_rnn_layers": 1,
            "num_classes": 1025,
        }
    },
    "decoding": {"model_path": "/must/not/read/external/tokenizer.model"},
}
type Node = tuple[str, str, list[int | str | None]]
SIGNATURES: dict[str, tuple[list[Node], list[Node]]] = {
    "encoder": (
        [
            ("audio_signal", "tensor(float)", ["batch", 64, "time"]),
            ("length", "tensor(int64)", ["batch"]),
        ],
        [
            ("encoded", "tensor(float)", ["batch", 768, "encoded_time"]),
            ("encoded_len", "tensor(int32)", ["batch"]),
        ],
    ),
    "decoder": (
        [
            ("x", "tensor(int64)", ["batch", 1]),
            ("hi", "tensor(float)", [1, "batch", 320]),
            ("ci", "tensor(float)", [1, "batch", 320]),
        ],
        [
            ("dec", "tensor(float)", ["batch", 1, 320]),
            ("ho", "tensor(float)", [1, "batch", 320]),
            ("co", "tensor(float)", [1, "batch", 320]),
        ],
    ),
    "joint": (
        [
            ("enc", "tensor(float)", ["batch", 768, 1]),
            ("dec", "tensor(float)", ["batch", 320, 1]),
        ],
        [("joint", "tensor(float)", ["batch", 1, 1, 1025])],
    ),
}


class FakeSession:
    def __init__(self, role, runtime):
        self.role = role
        self.runtime = runtime
        self.inputs, self.outputs = [
            [SimpleNamespace(name=n, type=t, shape=s) for n, t, s in reversed(nodes)]
            for nodes in runtime.signatures[role]
        ]

    def get_inputs(self):
        self.runtime.validated.add(id(self))
        return self.inputs

    def get_outputs(self):
        self.runtime.validated.add(id(self))
        return self.outputs

    def run(self, names, feeds):
        if not self.runtime.allow_inference:
            raise AssertionError("Loader must not run inference")
        calls = self.runtime.runs[self.role]
        calls.append((id(self), names, feeds))
        count = len(calls)
        if self.role == "encoder":
            assert set(feeds) == {"audio_signal", "length"}
            assert feeds["audio_signal"].shape[1] == 64
            assert feeds["length"].dtype == np.int64
            values = {
                "encoded": np.full((1, 768, 2), 11, dtype=np.float32),
                "encoded_len": np.array([2], dtype=np.int32),
            }
        elif self.role == "decoder":
            assert set(feeds) == {"x", "hi", "ci"}
            assert feeds["x"].shape == (1, 1)
            assert feeds["x"].dtype == np.int64
            assert feeds["hi"].shape == feeds["ci"].shape == (1, 1, 320)
            values = {
                "dec": np.full((1, 1, 320), 20 + count, dtype=np.float32),
                "ho": np.full((1, 1, 320), 100 + count, dtype=np.float32),
                "co": np.full((1, 1, 320), 200 + count, dtype=np.float32),
            }
        else:
            assert set(feeds) == {"enc", "dec"}
            np.testing.assert_array_equal(feeds["enc"], np.full((1, 768, 1), 11))
            np.testing.assert_array_equal(
                feeds["dec"], np.full((1, 320, 1), 20 + count)
            )
            logits = np.zeros((1, 1, 1, 1025), dtype=np.float32)
            logits[0, 0, 0, self.runtime.tokens[count - 1]] = 1
            values = {"joint": logits}
        # Respect requested output names, exactly as ORT does.
        return [values[name] for name in names]


class FakeTokenizer:
    def __init__(self, runtime):
        self.runtime = runtime

    def load(self, path):
        self.runtime.tokenizer_paths.append(Path(path))
        if self.runtime.fail == "tokenizer_load":
            raise RuntimeError("synthetic tokenizer load failure")
        return self.runtime.tokenizer_load

    def __len__(self):
        return self.runtime.vocab_size

    def decode(self, tokens):
        self.runtime.decoded.append((id(self), tokens))
        return " ".join(str(token) for token in tokens)


class FakeRuntime:
    def __init__(self):
        self.signatures = deepcopy(SIGNATURES)
        self.refs = []
        self.session_calls = []
        self.tokenizer_paths = []
        self.validated = set()
        self.runs = {role: [] for role in SIGNATURES}
        self.decoded = []
        self.allow_inference = False
        self.tokens = [7, 1024, 8, 1024]
        self.vocab_size = 1024
        self.tokenizer_load = True
        self.fail = None

    def session(self, path, *, providers, sess_options):
        role = Path(path).stem.rsplit("_", 1)[1]
        self.session_calls.append((Path(path), providers, sess_options))
        if self.fail == role:
            raise RuntimeError(f"synthetic {role} failure")
        result = FakeSession(role, self)
        self.refs.append(weakref.ref(result))
        return result

    def tokenizer(self):
        if self.fail == "tokenizer":
            raise RuntimeError("synthetic tokenizer factory failure")
        result = FakeTokenizer(self)
        self.refs.append(weakref.ref(result))
        return result

    @contextmanager
    def installed(self):
        ort = SimpleNamespace(
            __version__="fake-ort",
            SessionOptions=SimpleNamespace,
            GraphOptimizationLevel=SimpleNamespace(ORT_ENABLE_ALL="all"),
            ExecutionMode=SimpleNamespace(ORT_SEQUENTIAL="sequential"),
            InferenceSession=self.session,
        )
        sp = SimpleNamespace(
            __version__="fake-sp", SentencePieceProcessor=self.tokenizer
        )
        with patch.dict("sys.modules", {"onnxruntime": ort, "sentencepiece": sp}):
            yield self


@contextmanager
def fresh_ort_import(runtime, *, missing=False):
    """Observe the real import machinery before any ORT module can execute."""
    assert "onnxruntime" not in sys.modules, "Use a fresh child process"
    observed = []
    with runtime.installed():
        fake_ort = sys.modules.pop("onnxruntime")

        class Guard(MetaPathFinder, Loader):
            def find_spec(self, fullname, path=None, target=None):
                if fullname != "onnxruntime":
                    return None
                value = os.environ.get("ORT_DISABLE_TELEMETRY")
                assert value == "1", f"ORT first import saw {value!r}, expected '1'"
                observed.append(value)
                if missing:
                    raise ModuleNotFoundError(
                        "No module named 'onnxruntime'", name="onnxruntime"
                    )
                return spec_from_loader(fullname, self)

            def create_module(self, spec):
                return None

            def exec_module(self, module):
                module.__dict__.update(vars(fake_ort))

        guard = Guard()
        sys.meta_path.insert(0, guard)
        try:
            yield observed
        finally:
            sys.meta_path.remove(guard)


class TelemetryImportTest(unittest.TestCase):
    def test_loader_opts_out_before_first_runtime_import(self):
        script = """
import sys, tempfile
from pathlib import Path
import yaml
from local_transcriber import model_validation as mv
from tests.test_model_validation import CONFIG, FakeRuntime, fresh_ort_import
missing = sys.argv[1] == 'missing'
runtime = FakeRuntime()
with tempfile.TemporaryDirectory() as directory:
    root = Path(directory)
    for file in mv.BUNDLE.files:
        (root / file.name).write_bytes(b'synthetic model')
    (root / 'v3_e2e_rnnt.yaml').write_text(yaml.safe_dump(CONFIG))
    with fresh_ort_import(runtime, missing=missing) as observed:
        try:
            with mv.load_model_directory(root) as loaded:
                assert not missing
                assert loaded.metadata.verification == 'unverified'
        except ModuleNotFoundError as error:
            assert missing and error.name == 'onnxruntime'
    assert observed == ['1']
    assert len(runtime.session_calls) == (0 if missing else 3)
    assert len(runtime.tokenizer_paths) == (0 if missing else 1)
    assert not any(runtime.runs.values())
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


class ModelValidationTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name).resolve()
        self.env = patch.dict(os.environ, {"HOME": str(self.root)}, clear=True)
        self.env.start()
        self.addCleanup(self.env.stop)
        self.model = self.bundle(self.root / "manual")
        self.runtime = FakeRuntime()
        self.runtime_patch = self.runtime.installed()
        self.runtime_patch.__enter__()
        self.addCleanup(self.runtime_patch.__exit__, None, None, None)

    def bundle(self, path):
        path.mkdir(parents=True)
        for file in mv.BUNDLE.files:
            (path / file.name).write_bytes(f"synthetic {file.name}".encode())
        (path / "v3_e2e_rnnt.yaml").write_text(yaml.safe_dump(CONFIG))
        return path

    def test_specification_matches_documented_trust_anchors(self):
        self.assertEqual(mv.BUNDLE.archive_size_bytes, 213643430)
        self.assertEqual(
            mv.BUNDLE.archive_sha256,
            "e5a75ab56ab6d3f3a70ab17dd1ce858fe8180597963839dc806014447483224c",
        )
        self.assertEqual(
            [f.size_bytes for f in mv.BUNDLE.files],
            [319200791, 4600282, 687938, 255336, 1031],
        )
        self.assertEqual(
            [f.sha256 for f in mv.BUNDLE.files],
            [
                "dbea5c6158413e34b3707b99b65ec394c63cbd32da4164311f86dd65f86563d6",
                "a0f27fd86246d57cbe2c7138f3355591fd039e3e1015fd4ff2fd2d3d2d4d319d",
                "8bf573aca80d99ca4226aa0f9d43398998280ec505705955b5c6d92238b3e6d5",
                "828c12c991019eef952a960661f25a92d6ad279591e2ea466b4aeddf1d20a18a",
                "347037fe1858939df203f7195a36ad6d20995a3d8d3ae736cf501b082a246967",
            ],
        )

    def test_resolver_priority_and_normalization(self):
        legacy = self.bundle(self.root / ".giga/model")
        self.assertEqual(mv.resolve_model_dir(), legacy)
        cache = self.bundle(
            self.root / ".cache/local-transcriber/models" / "gigaam-v3-e2e-rnnt-int8-v1"
        )
        self.assertEqual(mv.resolve_model_dir(), cache)
        with patch.dict(os.environ, {"XDG_CACHE_HOME": str(self.root / "xdg")}):
            self.assertEqual(mv.resolve_model_dir(), legacy)
            xdg_cache = self.bundle(
                self.root
                / "xdg/local-transcriber/models"
                / "gigaam-v3-e2e-rnnt-int8-v1"
            )
            self.assertEqual(mv.resolve_model_dir(), xdg_cache)
        with patch.dict(os.environ, {"GIGAAM_MODEL_DIR": "~/manual"}):
            self.assertEqual(mv.resolve_model_dir(), self.model)
            self.assertEqual(mv.resolve_model_dir(legacy), legacy)
        link = self.root / "link"
        link.symlink_to(self.model, target_is_directory=True)
        self.assertEqual(mv.resolve_model_dir(link / ".." / "link"), self.model)
        self.assertEqual(mv.resolve_model_dir(Path("~/manual")), self.model)
        relative = Path(os.path.relpath(self.model))
        self.assertEqual(mv.resolve_model_dir(relative), self.model)
        with mv.load_model(legacy) as loaded:
            self.assertEqual(loaded.metadata.verification, "unverified")

    def test_bad_explicit_and_env_never_fall_back(self):
        self.bundle(self.root / ".giga/model")
        self.bundle(mv.managed_cache_dir())
        incomplete = self.root / "incomplete"
        incomplete.mkdir()
        file_path = self.root / "file"
        file_path.write_text("not a directory")
        for path in (self.root / "missing", incomplete, file_path):
            with (
                self.subTest(path=path),
                patch.dict(os.environ, {"GIGAAM_MODEL_DIR": str(self.model)}),
                self.assertRaises(mv.ModelValidationError),
            ):
                mv.load_model(path)
            with (
                self.subTest(env=path),
                patch.dict(os.environ, {"GIGAAM_MODEL_DIR": str(path)}),
                self.assertRaises(mv.ModelValidationError),
            ):
                mv.load_model()
        with (
            patch.dict(os.environ, {"GIGAAM_MODEL_DIR": ""}),
            self.assertRaisesRegex(mv.ModelValidationError, "must not be empty"),
        ):
            mv.load_model()
        self.assertEqual(self.runtime.session_calls, [])

    def test_absent_model_has_offline_install_instruction(self):
        with self.assertRaisesRegex(mv.ModelValidationError, "explicitly install"):
            mv.load_model()
        self.assertFalse(mv.managed_cache_dir().exists())

    def test_corrupt_cache_stops_search_even_with_valid_legacy(self):
        self.bundle(self.root / ".giga/model")
        cache = mv.managed_cache_dir()
        cache.parent.mkdir(parents=True)
        cache.symlink_to(self.root / "missing")
        with self.assertRaises(mv.ModelValidationError):
            mv.load_model()
        cache.unlink()
        self.bundle(cache)
        (cache / mv.MANIFEST_NAME).write_text(json.dumps(self.manifest()))
        encoder = cache / "v3_e2e_rnnt_encoder.onnx"
        original = encoder.read_bytes()
        encoder.write_bytes(b"X" + original[1:])
        with self.assertRaisesRegex(mv.ModelValidationError, "integrity mismatch"):
            mv.load_model()
        encoder.write_bytes(original)
        (cache / mv.MANIFEST_NAME).write_text("{broken")
        with self.assertRaisesRegex(mv.ModelValidationError, "manifest"):
            mv.load_model()
        (cache / mv.MANIFEST_NAME).unlink()
        (cache / "v3_e2e_rnnt.yaml").write_text("preprocessor: {}")
        with self.assertRaisesRegex(mv.ModelValidationError, "YAML"):
            mv.load_model()
        self.assertEqual(self.runtime.session_calls, [])

    def test_explicit_staging_api_never_uses_resolver(self):
        with (
            patch.dict(os.environ, {"GIGAAM_MODEL_DIR": str(self.model)}),
            patch.object(
                mv, "resolve_model_dir", side_effect=AssertionError("no resolution")
            ),
        ):
            with mv.load_model_directory(self.model) as model:
                metadata = model.metadata
                self.assertEqual(metadata.model_dir, self.model)
            self.assertEqual(model.encoder, None)
            self.assertEqual(metadata.verification, "unverified")
            with self.assertRaises(mv.ModelValidationError):
                mv.load_model_directory(self.root / "absent-staging")
        gc.collect()
        self.assertTrue(all(ref() is None for ref in self.runtime.refs))

    def test_missing_nonregular_and_unreadable_files_fail_before_sessions(self):
        for file in mv.BUNDLE.files:
            path = self.model / file.name
            original = path.read_bytes()
            path.unlink()
            with (
                self.subTest(missing=file.name),
                self.assertRaisesRegex(mv.ModelValidationError, "model file"),
            ):
                mv.load_model_directory(self.model)
            path.mkdir()
            with (
                self.subTest(directory=file.name),
                self.assertRaisesRegex(mv.ModelValidationError, "regular"),
            ):
                mv.load_model_directory(self.model)
            path.rmdir()
            path.write_bytes(original)
        original_open = Path.open

        def denied(path, *args, **kwargs):
            if path.name == "v3_e2e_rnnt_encoder.onnx":
                raise PermissionError("synthetic permission failure")
            return original_open(path, *args, **kwargs)

        with (
            patch.object(Path, "open", denied),
            self.assertRaisesRegex(mv.ModelValidationError, "Cannot read"),
        ):
            mv.load_model_directory(self.model)
        self.assertEqual(self.runtime.session_calls, [])

    def test_strict_yaml_rejects_missing_wrong_and_unsafe_values(self):
        path = self.model / "v3_e2e_rnnt.yaml"
        for section, keys in (
            ("preprocessor", CONFIG["preprocessor"]),
            ("decoder", CONFIG["head"]["decoder"]),
        ):
            for key in keys:
                for mutation in ("missing", "wrong-type", "wrong-value"):
                    config = deepcopy(CONFIG)
                    target = (
                        config[section]
                        if section == "preprocessor"
                        else config["head"][section]
                    )
                    if mutation == "missing":
                        del target[key]
                    else:
                        target[key] = "incompatible" if mutation == "wrong-type" else 2
                    path.write_text(yaml.safe_dump(config))
                    with (
                        self.subTest(section=section, key=key, mutation=mutation),
                        self.assertRaisesRegex(mv.ModelValidationError, "YAML"),
                    ):
                        mv.load_model_directory(self.model)
        for key, value in (("center", 0), ("sample_rate", 16000.0)):
            config = deepcopy(CONFIG)
            config["preprocessor"][key] = value
            path.write_text(yaml.safe_dump(config))
            with (
                self.subTest(key=key, value=value),
                self.assertRaisesRegex(mv.ModelValidationError, "YAML"),
            ):
                mv.load_model_directory(self.model)
        for text in (
            "[]",
            "null",
            "head: [",
            "!!python/object/apply:os.system ['false']",
            "preprocessor: {}\npreprocessor: {}",
            "{[a, b]: c}",
        ):
            path.write_text(text)
            with self.subTest(text=text), self.assertRaises(mv.ModelValidationError):
                mv.load_model_directory(self.model)
        config = deepcopy(CONFIG)
        config["preprocessor"]["channels"] = 2
        path.write_text(yaml.safe_dump(config))
        with self.assertRaisesRegex(mv.ModelValidationError, "mono"):
            mv.load_model_directory(self.model)
        self.assertEqual(self.runtime.session_calls, [])

    def test_named_signatures_reject_wrong_name_dtype_rank_and_dimensions(self):
        for role, directions in SIGNATURES.items():
            for direction, nodes in enumerate(directions):
                for index, (name, dtype, shape) in enumerate(nodes):
                    bad_shape = list(shape)
                    bad_shape[-1] = 999
                    variants = [
                        ("wrong_name", dtype, shape),
                        (name, "tensor(double)", shape),
                        (name, dtype, [*shape, 1]),
                        (name, dtype, bad_shape),
                    ]
                    for node in variants:
                        self.runtime.signatures = deepcopy(SIGNATURES)
                        self.runtime.signatures[role][direction][index] = node
                        with (
                            self.subTest(role=role, direction=direction, node=node),
                            self.assertRaisesRegex(
                                mv.ModelValidationError, "ONNX|dtype/rank|dimension"
                            ),
                        ):
                            mv.load_model_directory(self.model)
        self.runtime.signatures = deepcopy(SIGNATURES)
        self.runtime.signatures["encoder"][1][1] = (
            "encoded_len",
            "tensor(int64)",
            ["B"],
        )
        with self.assertRaisesRegex(mv.ModelValidationError, "encoded_len"):
            mv.load_model_directory(self.model)
        self.runtime.signatures = deepcopy(SIGNATURES)
        self.runtime.signatures["joint"][0].append(
            self.runtime.signatures["joint"][0][0]
        )
        with self.assertRaisesRegex(mv.ModelValidationError, "names"):
            mv.load_model_directory(self.model)

    def test_symbolic_names_and_static_batch_one_are_compatible(self):
        for directions in self.runtime.signatures.values():
            for nodes in directions:
                for _, _, shape in nodes:
                    for i, dim in enumerate(shape):
                        if dim == "batch":
                            shape[i] = 1
                        elif isinstance(dim, str):
                            shape[i] = "arbitrary_symbol"
        with mv.load_model_directory(self.model) as model:
            self.assertEqual(model.metadata.verification, "unverified")

    def test_tokenizer_load_and_class_compatibility(self):
        for size in (0, 1023, 1025):
            self.runtime.vocab_size = size
            with (
                self.subTest(size=size),
                self.assertRaisesRegex(mv.ModelValidationError, "size/blank/classes"),
            ):
                mv.load_model_directory(self.model)
        self.runtime.vocab_size = 1024
        self.runtime.tokenizer_load = False
        with self.assertRaisesRegex(
            mv.ModelValidationError, "tokenizer could not be loaded"
        ):
            mv.load_model_directory(self.model)
        self.assertTrue(
            all(
                p == self.model / "v3_e2e_rnnt_tokenizer.model"
                for p in self.runtime.tokenizer_paths
            )
        )

    def manifest(self):
        return {
            "schema_version": 1,
            "profile": "v3_e2e_rnnt",
            "bundle_sha256": mv.BUNDLE.archive_sha256,
            "versions": {"installer": "synthetic-test"},
            "files": {
                p.name: {
                    "size_bytes": len(p.read_bytes()),
                    "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                }
                for p in self.model.iterdir()
                if p.name != mv.MANIFEST_NAME
            },
        }

    def synthetic_spec(self):
        # Explicit test trust root, NEVER a claim that fake weights match production.
        return replace(
            mv.BUNDLE,
            files=tuple(
                mv.FileSpec(
                    f.name,
                    len((self.model / f.name).read_bytes()),
                    hashlib.sha256((self.model / f.name).read_bytes()).hexdigest(),
                )
                for f in mv.BUNDLE.files
            ),
        )

    def test_self_consistent_manifest_cannot_create_pinned_trust(self):
        path = self.model / mv.MANIFEST_NAME
        path.write_text(json.dumps(self.manifest()))
        with mv.load_model_directory(self.model) as loaded:
            self.assertTrue(loaded.metadata.manifest_integrity)
            self.assertEqual(loaded.metadata.verification, "unverified")
            self.assertNotIn("bundle_sha256", loaded.metadata.model_identity())
            with self.assertRaisesRegex(mv.ModelValidationError, "Only pinned"):
                loaded.metadata.install_manifest()

    def test_corrupt_manifest_never_downgrades_to_manual(self):
        valid = self.manifest()
        corruptions = [
            [],
            {},
            {**valid, "schema_version": True},
            {**valid, "schema_version": 2},
            {**valid, "profile": "other"},
            {**valid, "bundle_sha256": "invalid"},
            {**valid, "files": {}},
            {**valid, "versions": {}},
            {**valid, "versions": {"ort": 123}},
        ]
        for key, value in (("sha256", "0" * 64), ("size_bytes", 999)):
            corrupt = deepcopy(valid)
            corrupt["files"]["v3_e2e_rnnt_encoder.onnx"][key] = value
            corruptions.append(corrupt)
        path = self.model / mv.MANIFEST_NAME
        for content in corruptions:
            path.write_text(json.dumps(content))
            with (
                self.subTest(content=content),
                self.assertRaises(mv.ModelValidationError),
            ):
                mv.load_model_directory(self.model)
        path.write_text('{"schema_version": 1, "schema_version": 1}')
        with self.assertRaisesRegex(mv.ModelValidationError, "Duplicate"):
            mv.load_model_directory(self.model)
        path.unlink()
        path.symlink_to(self.root / "absent-manifest")
        with self.assertRaisesRegex(mv.ModelValidationError, "manifest"):
            mv.load_model_directory(self.model)
        self.assertEqual(self.runtime.session_calls, [])

    def test_pinned_requires_all_five_trusted_files_and_reuses_hashes(self):
        spec = self.synthetic_spec()
        with patch.object(mv, "BUNDLE", spec):
            with mv.load_model_directory(self.model) as loaded:
                manifest = loaded.metadata.install_manifest()
                self.assertEqual(loaded.metadata.verification, "pinned")
            (self.model / mv.MANIFEST_NAME).write_text(json.dumps(manifest))
            with patch.object(mv, "hash_file", wraps=mv.hash_file) as hashes:
                with mv.load_model_directory(self.model) as loaded:
                    identity = loaded.metadata.model_identity()
                    self.assertEqual(identity["bundle_sha256"], spec.archive_sha256)
                    self.assertEqual(
                        identity["file_sha256"], {f.name: f.sha256 for f in spec.files}
                    )
                    self.assertTrue(loaded.metadata.manifest_integrity)
                    self.assertEqual(loaded.metadata.install_manifest(), manifest)
                self.assertEqual(
                    [c.args[0] for c in hashes.call_args_list],
                    [self.model / f.name for f in spec.files],
                )
            (self.model / mv.MANIFEST_NAME).unlink()
            for file in spec.files:
                path = self.model / file.name
                data = path.read_bytes()
                path.write_bytes(
                    data + (b"\n# edited\n" if path.suffix == ".yaml" else b"edited")
                )
                with (
                    self.subTest(file=file.name),
                    mv.load_model_directory(self.model) as changed,
                ):
                    self.assertEqual(changed.metadata.verification, "unverified")
                    self.assertNotIn("bundle_sha256", changed.metadata.model_identity())
                path.write_bytes(data)

    def test_streaming_hash_reads_once_with_bounded_blocks(self):
        path = self.model / "v3_e2e_rnnt_encoder.onnx"
        data = b"synthetic" * (mv.HASH_BLOCK_BYTES // 3)
        path.write_bytes(data)
        reads = []
        original_open = Path.open

        @contextmanager
        def tracked_open(file, mode="r", *args, **kwargs):
            with original_open(file, mode, *args, **kwargs) as stream:
                if mode == "rb":

                    class Reader:
                        def read(self, count):
                            reads.append((file, count))
                            return stream.read(count)

                    yield Reader()
                else:
                    yield stream

        with (
            patch.object(Path, "open", tracked_open),
            mv.load_model_directory(self.model) as loaded,
        ):
            self.assertEqual(
                loaded.metadata.files[0].sha256, hashlib.sha256(data).hexdigest()
            )
            loaded.metadata.model_identity()
        expected_reads = len(data) // mv.HASH_BLOCK_BYTES + 2
        self.assertEqual(sum(file == path for file, _ in reads), expected_reads)
        self.assertTrue(all(size == mv.HASH_BLOCK_BYTES for _, size in reads))
        for file in mv.BUNDLE.files[1:]:
            self.assertEqual(sum(p.name == file.name for p, _ in reads), 2)

    def test_one_owned_runtime_named_io_and_greedy_state_semantics(self):
        with patch.object(mv, "hash_file", wraps=mv.hash_file) as hashes:
            loaded = mv.load_model_directory(self.model, threads=2)
            objects = loaded.encoder, loaded.decoder, loaded.joint, loaded.tokenizer
            engine = GigaAMEngine(loaded_model=loaded)
            self.assertEqual(hashes.call_count, 5)
            self.assertEqual(len(self.runtime.session_calls), 3)
            self.assertEqual(len(self.runtime.tokenizer_paths), 1)
            for actual, expected in zip(
                (engine.encoder, engine.decoder, engine.joint, engine.tokenizer),
                objects,
                strict=True,
            ):
                self.assertIs(actual, expected)
            self.assertEqual(self.runtime.validated, {id(obj) for obj in objects[:3]})
            self.assertIsNone(loaded.encoder)
            with self.assertRaisesRegex(mv.ModelValidationError, "transferred"):
                loaded.take_runtime()
            self.runtime.allow_inference = True
            self.assertEqual(
                engine.transcribe_wave(np.zeros(640, dtype=np.float32)), "7 8"
            )
            self.assertEqual(hashes.call_count, 5)
        calls = self.runtime.runs["decoder"]
        self.assertEqual([int(c[2]["x"][0, 0]) for c in calls], [1024, 7, 7, 8])
        self.assertEqual(
            [float(c[2]["hi"][0, 0, 0]) for c in calls], [0, 101, 101, 103]
        )
        self.assertEqual(
            [float(c[2]["ci"][0, 0, 0]) for c in calls], [0, 201, 201, 203]
        )
        self.assertEqual(self.runtime.decoded, [(id(engine.tokenizer), [7, 8])])
        for role in SIGNATURES:
            self.assertEqual(
                {c[0] for c in self.runtime.runs[role]}, {id(getattr(engine, role))}
            )
        for path, providers, options in self.runtime.session_calls:
            self.assertEqual(path.parent, self.model)
            self.assertEqual(providers, ["CPUExecutionProvider"])
            self.assertEqual(options.intra_op_num_threads, 2)
            self.assertEqual(options.execution_mode, "sequential")
        # Dropping the engine is sufficient; the transferred LoadedModel retains none.
        del objects, actual, expected, engine
        gc.collect()
        self.assertTrue(all(ref() is None for ref in self.runtime.refs))

    def test_max_three_symbols_per_frame_is_unchanged(self):
        engine = GigaAMEngine(self.model)
        self.runtime.allow_inference = True
        self.runtime.tokens = [1, 2, 3, 4, 5, 6]
        self.assertEqual(
            engine.transcribe_wave(np.zeros(640, dtype=np.float32)), "1 2 3 4 5 6"
        )
        self.assertEqual(len(self.runtime.runs["joint"]), 6)

    def test_failure_releases_partial_runtime_even_if_exception_is_kept(self):
        errors = []
        for failure in ("encoder", "decoder", "joint", "tokenizer", "tokenizer_load"):
            self.runtime.fail = failure
            try:
                mv.load_model_directory(self.model)
            except mv.ModelValidationError as error:
                errors.append(error)
            else:
                self.fail("expected model load failure")
            gc.collect()
            self.assertTrue(all(ref() is None for ref in self.runtime.refs), failure)
        self.runtime.fail = None
        self.runtime.vocab_size = 2
        try:
            mv.load_model_directory(self.model)
        except mv.ModelValidationError as error:
            errors.append(error)
        gc.collect()
        self.assertTrue(all(ref() is None for ref in self.runtime.refs))
        self.runtime.vocab_size = 1024
        self.runtime.signatures["decoder"][0][0] = ("wrong", "tensor(int64)", ["B", 1])
        try:
            mv.load_model_directory(self.model)
        except mv.ModelValidationError as error:
            errors.append(error)
        gc.collect()
        self.assertTrue(all(ref() is None for ref in self.runtime.refs))
        self.assertEqual(len(errors), 7)

    def test_cli_metadata_uses_real_loader_and_shared_outputs_for_both_trust_states(
        self,
    ):
        source = self.root / "synthetic.wav"
        source.write_bytes(b"not read: audio boundary stubbed")
        raw = TranscriptResult(
            [
                {"start": 0, "end": 1, "text": "Subtitles by fixture"},
                {"start": 1, "end": 24, "text": "first"},
                {"start": 24, "end": 25, "text": "Subtitles by fixture"},
                {"start": 25, "end": 49, "text": "second"},
            ],
            "ru",
            55.0,
        )
        for pinned in (False, True):
            spec = self.synthetic_spec() if pinned else mv.BUNDLE
            out = self.root / f"output-{pinned}"
            with (
                patch.object(mv, "BUNDLE", spec),
                patch.object(GigaAMEngine, "transcribe_result", return_value=raw),
                patch.object(mv, "hash_file", wraps=mv.hash_file) as hashes,
                redirect_stdout(io.StringIO()),
            ):
                cli.main(
                    [
                        str(source),
                        "--engine",
                        "gigaam",
                        "--gigaam-model-dir",
                        str(self.model),
                        "--out-dir",
                        str(out),
                        "--drop-subtitle-artifacts",
                    ]
                )
                self.assertEqual(hashes.call_count, 5)
            payload = json.loads(next(out.rglob("transcript.json")).read_bytes())
            identity = payload["model_identity"]
            self.assertEqual(
                identity["verification"], "pinned" if pinned else "unverified"
            )
            self.assertEqual(identity["profile"], "v3_e2e_rnnt")
            self.assertEqual(identity["model_dir"], str(self.model))
            self.assertEqual(
                identity["file_sha256"],
                {
                    name: entry["sha256"]
                    for name, entry in self.manifest()["files"].items()
                },
            )
            if pinned:
                self.assertEqual(identity["bundle_sha256"], spec.archive_sha256)
            else:
                self.assertNotIn("bundle_sha256", identity)
            self.assertEqual(payload["gigaam_profile"], identity["profile"])
            self.assertEqual(payload["gigaam_model_dir"], identity["model_dir"])
            self.assertEqual(payload["artifact_type"], "transcript")
            self.assertEqual(payload["schema_version"], 1)
            self.assertEqual(payload["view_raw_indices"], [[1], [3]])
            self.assertEqual(payload["processing"]["merge_policy"], "none")
            self.assertEqual(payload["duration_seconds"], 55.0)
            self.assertEqual(payload["raw_segments"], raw.raw_segments)
            self.assertEqual(payload["device"], "cpu")
            self.assertEqual(payload["requested_device"], "auto")

    def test_bad_model_cli_is_short_error_without_asr_or_whisper(self):
        source = self.root / "synthetic.wav"
        source.write_bytes(b"must not be decoded")
        (self.model / mv.MANIFEST_NAME).write_text("broken")
        stderr = io.StringIO()
        with (
            patch.object(GigaAMEngine, "transcribe_result") as asr,
            patch.object(cli, "transcribe") as whisper,
            redirect_stderr(stderr),
            redirect_stdout(io.StringIO()),
            self.assertRaises(SystemExit) as raised,
        ):
            cli.main(
                [
                    str(source),
                    "--engine",
                    "gigaam",
                    "--gigaam-model-dir",
                    str(self.model),
                    "--out-dir",
                    str(self.root / "outputs"),
                ]
            )
        self.assertEqual(raised.exception.code, 2)
        self.assertIn("manifest", stderr.getvalue())
        self.assertNotIn("Traceback", stderr.getvalue())
        self.assertEqual(self.runtime.session_calls, [])
        asr.assert_not_called()
        whisper.assert_not_called()
        self.assertEqual(list((self.root / "outputs").rglob("transcript.*")), [])


if __name__ == "__main__":
    unittest.main()
