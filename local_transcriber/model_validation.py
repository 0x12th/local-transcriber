"""Offline GigaAM specification, path selection and single-owner CPU loading.

Structural compatibility is not model provenance or a sandbox. Only the trusted
file digests below identify the pinned bundle; a local manifest proves neither.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from local_transcriber.runtime_policy import disable_ort_telemetry


@dataclass(frozen=True)
class FileSpec:
    name: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True)
class TensorSpec:
    name: str
    dtype: str
    shape: tuple[int | str, ...]


@dataclass(frozen=True)
class GraphSpec:
    role: str
    inputs: tuple[TensorSpec, ...]
    outputs: tuple[TensorSpec, ...]


@dataclass(frozen=True)
class BundleSpec:
    profile: str
    cache_name: str
    archive_url: str
    archive_size_bytes: int
    archive_sha256: str
    files: tuple[FileSpec, ...]
    graphs: tuple[GraphSpec, ...]


BUNDLE = BundleSpec(
    profile="v3_e2e_rnnt",
    cache_name="gigaam-v3-e2e-rnnt-int8-v1",
    archive_url=(
        "https://github.com/moznoazachem/giga-pisar-cli/releases/download/"
        "v1.0/gigaam-v3-onnx-int8.tar.gz"
    ),
    archive_size_bytes=213643430,
    archive_sha256="e5a75ab56ab6d3f3a70ab17dd1ce858fe8180597963839dc806014447483224c",
    files=(
        FileSpec(
            "v3_e2e_rnnt_encoder.onnx",
            319200791,
            "dbea5c6158413e34b3707b99b65ec394c63cbd32da4164311f86dd65f86563d6",
        ),
        FileSpec(
            "v3_e2e_rnnt_decoder.onnx",
            4600282,
            "a0f27fd86246d57cbe2c7138f3355591fd039e3e1015fd4ff2fd2d3d2d4d319d",
        ),
        FileSpec(
            "v3_e2e_rnnt_joint.onnx",
            687938,
            "8bf573aca80d99ca4226aa0f9d43398998280ec505705955b5c6d92238b3e6d5",
        ),
        FileSpec(
            "v3_e2e_rnnt_tokenizer.model",
            255336,
            "828c12c991019eef952a960661f25a92d6ad279591e2ea466b4aeddf1d20a18a",
        ),
        FileSpec(
            "v3_e2e_rnnt.yaml",
            1031,
            "347037fe1858939df203f7195a36ad6d20995a3d8d3ae736cf501b082a246967",
        ),
    ),
    graphs=(
        GraphSpec(
            "encoder",
            (
                TensorSpec("audio_signal", "tensor(float)", ("B", 64, "T")),
                TensorSpec("length", "tensor(int64)", ("B",)),
            ),
            (
                TensorSpec("encoded", "tensor(float)", ("B", 768, "T")),
                TensorSpec("encoded_len", "tensor(int32)", ("B",)),
            ),
        ),
        GraphSpec(
            "decoder",
            (
                TensorSpec("x", "tensor(int64)", ("B", 1)),
                TensorSpec("hi", "tensor(float)", (1, "B", 320)),
                TensorSpec("ci", "tensor(float)", (1, "B", 320)),
            ),
            (
                TensorSpec("dec", "tensor(float)", ("B", 1, 320)),
                TensorSpec("ho", "tensor(float)", (1, "B", 320)),
                TensorSpec("co", "tensor(float)", (1, "B", 320)),
            ),
        ),
        GraphSpec(
            "joint",
            (
                TensorSpec("enc", "tensor(float)", ("B", 768, 1)),
                TensorSpec("dec", "tensor(float)", ("B", 320, 1)),
            ),
            (TensorSpec("joint", "tensor(float)", ("B", 1, 1, 1025)),),
        ),
    ),
)
MANIFEST_NAME = "model-manifest.json"
HASH_BLOCK_BYTES = 1024 * 1024


class ModelValidationError(ValueError):
    """The selected model is invalid; callers must not fall back to another one."""


def _directory(path: Path) -> Path:
    try:
        resolved = path.expanduser().resolve()
        if not resolved.is_dir():
            raise ModelValidationError(f"GigaAM model directory not found: {resolved}")
        return resolved
    except (OSError, RuntimeError) as error:
        raise ModelValidationError(
            f"Cannot resolve GigaAM model {path}: {error}"
        ) from error


def managed_cache_dir() -> Path:
    root = Path(os.environ.get("XDG_CACHE_HOME") or "~/.cache").expanduser()
    return root / "local-transcriber" / "models" / BUNDLE.cache_name


def resolve_model_dir(explicit: Path | None = None) -> Path:
    """Select once. Only an absent managed cache permits the legacy candidate."""
    if explicit is not None:
        return _directory(explicit)
    if "GIGAAM_MODEL_DIR" in os.environ:
        value = os.environ["GIGAAM_MODEL_DIR"]
        if not value.strip():
            raise ModelValidationError("GIGAAM_MODEL_DIR must not be empty")
        return _directory(Path(value))
    cache = managed_cache_dir()
    try:
        cache.lstat()  # Includes broken symlinks; inaccessible is not absent.
    except FileNotFoundError:
        pass
    else:
        return _directory(cache)
    legacy = Path.home() / ".giga" / "model"
    if legacy.exists() or legacy.is_symlink():
        return _directory(legacy)
    raise ModelValidationError(
        "GigaAM model not found; explicitly install the v3_e2e_rnnt ONNX bundle "
        "and pass --gigaam-model-dir or set GIGAAM_MODEL_DIR. "
        "Transcription never downloads models."
    )


def hash_file(path: Path) -> FileSpec:
    """One bounded streaming pass, reused for integrity and provenance."""
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        while block := stream.read(HASH_BLOCK_BYTES):
            size += len(block)
            digest.update(block)
    return FileSpec(path.name, size, digest.hexdigest())


def _unique_mapping(pairs: list[tuple[Any, Any]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ModelValidationError(f"Duplicate model metadata key: {key}")
        result[key] = value
    return result


def _read_config(path: Path) -> dict[str, Any]:
    import yaml
    from yaml.resolver import BaseResolver

    class UniqueSafeLoader(yaml.SafeLoader):
        pass

    def mapping(loader: Any, node: Any) -> dict:
        return _unique_mapping(loader.construct_pairs(node, deep=True))

    UniqueSafeLoader.add_constructor(BaseResolver.DEFAULT_MAPPING_TAG, mapping)
    try:
        with path.open(encoding="utf-8") as stream:
            config = yaml.load(stream, Loader=UniqueSafeLoader)
    except (yaml.YAMLError, UnicodeError, TypeError) as error:
        raise ModelValidationError(f"Invalid GigaAM YAML: {error}") from error
    expected = {
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
    }
    if not isinstance(config, dict):
        raise ModelValidationError("GigaAM YAML must be a mapping")
    decoder = config.get("head", {})
    decoder = decoder.get("decoder") if isinstance(decoder, dict) else None
    sections = [
        ("preprocessor", config.get("preprocessor"), expected["preprocessor"]),
        (
            "head.decoder",
            decoder,
            {"pred_hidden": 320, "pred_rnn_layers": 1, "num_classes": 1025},
        ),
    ]
    for name, section, fields in sections:
        if not isinstance(section, dict):
            raise ModelValidationError(f"GigaAM YAML requires {name}")
        for key, value in fields.items():
            if (
                key not in section
                or type(section[key]) is not type(value)
                or section[key] != value
            ):
                raise ModelValidationError(
                    f"GigaAM YAML requires {name}.{key}={value!r}"
                )
    # The published config need not declare channels; audio and ONNX are mono.
    preprocessor = config["preprocessor"]
    if "channels" in preprocessor and (
        type(preprocessor["channels"]) is not int or preprocessor["channels"] != 1
    ):
        raise ModelValidationError("GigaAM requires mono preprocessor.channels=1")
    return config


def _check_manifest(path: Path, files: tuple[FileSpec, ...], pinned: bool) -> bool:
    try:
        path.lstat()
    except FileNotFoundError:
        return False
    if not path.is_file():
        raise ModelValidationError(f"Invalid GigaAM manifest: {path}")
    try:
        manifest = json.loads(
            path.read_text(encoding="utf-8"), object_pairs_hook=_unique_mapping
        )
    except (ValueError, UnicodeError) as error:
        raise ModelValidationError(f"Invalid GigaAM manifest: {error}") from error
    if not isinstance(manifest, dict):
        raise ModelValidationError("GigaAM manifest must be an object")
    if (
        type(manifest.get("schema_version")) is not int
        or manifest["schema_version"] != 1
        or manifest.get("profile") != BUNDLE.profile
    ):
        raise ModelValidationError("Unsupported GigaAM manifest version/profile")
    archive_hash = manifest.get("bundle_sha256")
    if (
        not isinstance(archive_hash, str)
        or re.fullmatch(r"[0-9a-f]{64}", archive_hash) is None
    ):
        raise ModelValidationError("Invalid manifest bundle_sha256")
    if pinned and archive_hash != BUNDLE.archive_sha256:
        raise ModelValidationError("Manifest bundle_sha256 contradicts pinned files")
    versions = manifest.get("versions")
    if (
        not isinstance(versions, dict)
        or not versions
        or any(
            not isinstance(k, str) or not isinstance(v, str) or not v
            for k, v in versions.items()
        )
    ):
        raise ModelValidationError("Manifest requires versions")
    entries = manifest.get("files")
    if not isinstance(entries, dict) or set(entries) != {f.name for f in files}:
        raise ModelValidationError(
            "Manifest must describe exactly the five model files"
        )
    for file in files:
        entry = entries[file.name]
        if (
            not isinstance(entry, dict)
            or type(entry.get("size_bytes")) is not int
            or entry["size_bytes"] != file.size_bytes
            or entry.get("sha256") != file.sha256
        ):
            raise ModelValidationError(f"Manifest integrity mismatch: {file.name}")
    return True


def _check_tensors(
    nodes: list[Any], expected: tuple[TensorSpec, ...], label: str
) -> None:
    by_name = {node.name: node for node in nodes}
    if len(by_name) != len(nodes) or set(by_name) != {t.name for t in expected}:
        raise ModelValidationError(f"{label}: incompatible ONNX names")
    for tensor in expected:
        node = by_name[tensor.name]
        if (
            node.type != tensor.dtype
            or node.shape is None
            or len(node.shape) != len(tensor.shape)
        ):
            raise ModelValidationError(
                f"{label}.{tensor.name}: incompatible dtype/rank"
            )
        for actual, required in zip(node.shape, tensor.shape, strict=True):
            if isinstance(required, int):
                valid = type(actual) is int and actual == required
            elif required == "B":
                valid = actual is None or isinstance(actual, str) or actual == 1
            else:
                # Time must support arbitrary chunks, not a fixed export length.
                valid = actual is None or isinstance(actual, str)
            if not valid:
                raise ModelValidationError(
                    f"{label}.{tensor.name}: incompatible dimension {actual!r}"
                )


@dataclass(frozen=True)
class ModelMetadata:
    model_dir: Path
    files: tuple[FileSpec, ...]
    verification: Literal["pinned", "unverified"]
    manifest_integrity: bool
    versions: tuple[tuple[str, str], ...]

    def model_identity(self) -> dict[str, Any]:
        identity: dict[str, Any] = {
            "profile": BUNDLE.profile,
            "model_dir": str(self.model_dir),
            "verification": self.verification,
            "file_sha256": {f.name: f.sha256 for f in self.files},
        }
        if self.verification == "pinned":
            identity["bundle_sha256"] = BUNDLE.archive_sha256
        return identity

    def install_manifest(self) -> dict[str, Any]:
        """Facts for a future installer; does not write or publish anything."""
        if self.verification != "pinned":
            raise ModelValidationError(
                "Only pinned files can produce an install manifest"
            )
        return {
            "schema_version": 1,
            "profile": BUNDLE.profile,
            "bundle_sha256": BUNDLE.archive_sha256,
            "versions": dict(self.versions),
            "files": {
                f.name: {"size_bytes": f.size_bytes, "sha256": f.sha256}
                for f in self.files
            },
        }


@dataclass
class LoadedModel:
    """Owns checked objects until close() or a one-time transfer to an engine.

    Use as a context manager for staging validation. Metadata survives close;
    ORT has no public close API, so close drops this owner's runtime references.
    """

    metadata: ModelMetadata
    config: dict[str, Any]
    encoder: Any = None
    decoder: Any = None
    joint: Any = None
    tokenizer: Any = None

    def take_runtime(self) -> tuple[Any, Any, Any, Any]:
        if any(
            obj is None
            for obj in (self.encoder, self.decoder, self.joint, self.tokenizer)
        ):
            raise ModelValidationError(
                "Loaded GigaAM model is closed or already transferred"
            )
        objects = self.encoder, self.decoder, self.joint, self.tokenizer
        self.close()
        return objects

    def close(self) -> None:
        self.encoder = self.decoder = self.joint = self.tokenizer = None

    def __enter__(self) -> LoadedModel:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


def load_model_directory(model_dir: Path, threads: int = 0) -> LoadedModel:
    """Validate exactly this directory (including staging), never resolve/fallback.

    No inference, writes or downloads. Files must remain unchanged during loading
    and use; this API does not protect against concurrent hostile file replacement.
    """
    root = _directory(model_dir)
    if type(threads) is not int or threads < 0:
        raise ModelValidationError("GigaAM threads must be a non-negative integer")
    for file in BUNDLE.files:
        path = root / file.name
        try:
            if not stat.S_ISREG(path.stat().st_mode):
                raise ModelValidationError(f"Not a regular GigaAM model file: {path}")
        except OSError as error:
            raise ModelValidationError(
                f"Missing/unreadable GigaAM model file: {path}"
            ) from error
    try:
        files = tuple(hash_file(root / file.name) for file in BUNDLE.files)
        config = _read_config(root / f"{BUNDLE.profile}.yaml")
        pinned = files == BUNDLE.files
        integrity = _check_manifest(root / MANIFEST_NAME, files, pinned)
    except OSError as error:
        raise ModelValidationError(f"Cannot read GigaAM model: {error}") from error

    disable_ort_telemetry()
    import onnxruntime as ort
    import sentencepiece as sp
    import yaml

    versions = (
        ("onnxruntime", ort.__version__),
        ("sentencepiece", sp.__version__),
        ("pyyaml", yaml.__version__),
    )
    loaded = LoadedModel(
        ModelMetadata(
            root, files, "pinned" if pinned else "unverified", integrity, versions
        ),
        config,
    )
    # Do not retain native objects through chained exceptions/validation tracebacks.
    failure = None
    complete = False
    try:
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.intra_op_num_threads = threads or min(8, os.cpu_count() or 4)
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.log_severity_level = 3
        for graph in BUNDLE.graphs:
            setattr(
                loaded,
                graph.role,
                ort.InferenceSession(
                    str(root / f"{BUNDLE.profile}_{graph.role}.onnx"),
                    providers=["CPUExecutionProvider"],
                    sess_options=options,
                ),
            )
            _check_tensors(
                getattr(loaded, graph.role).get_inputs(),
                graph.inputs,
                f"{graph.role} inputs",
            )
            _check_tensors(
                getattr(loaded, graph.role).get_outputs(),
                graph.outputs,
                f"{graph.role} outputs",
            )
        loaded.tokenizer = sp.SentencePieceProcessor()
        if not loaded.tokenizer.load(str(root / f"{BUNDLE.profile}_tokenizer.model")):
            raise ModelValidationError("GigaAM tokenizer could not be loaded")
        if len(loaded.tokenizer) != config["head"]["decoder"]["num_classes"] - 1:
            raise ModelValidationError("GigaAM tokenizer size/blank/classes mismatch")
        complete = True
    except Exception as error:
        failure = f"Cannot load GigaAM model at {root}: {error}"
    finally:
        # BaseException must release owned objects without changing propagation.
        if not complete:
            loaded.close()
    if failure is not None:
        raise ModelValidationError(failure)
    return loaded


def load_model(explicit: Path | None = None, threads: int = 0) -> LoadedModel:
    """Runtime entry point: select a directory, then load it exactly once."""
    return load_model_directory(resolve_model_dir(explicit), threads)
