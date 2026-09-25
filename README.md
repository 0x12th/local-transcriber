# local-transcriber

Private, local audio and video transcription powered by
[`openai-whisper`](https://github.com/openai/whisper), with an optional local
GigaAM v3 ONNX backend. Whisper remains the default. The recording stays on
your machine; no hosted model or diarization service is used.

The CLI writes a plain Markdown transcript, a timestamped version, and
structured JSON. Whisper supports its usual languages and devices; the GigaAM
profile supported here is Russian-only and CPU-only.

## Requirements

- Python 3.12+
- [`uv`](https://docs.astral.sh/uv/)
- `ffmpeg` and `ffprobe` on `PATH`

Whisper and torch remain **mandatory dependencies**, including with `--extra gigaam`.
This is still a heavy download/disk installation, not GigaAM-only packaging.
GigaAM's lazy imports and lower runtime memory do not remove that installation cost.

## Online setup

Dependency setup and explicit model installation may use the network. From the
project root, prepare the environment and install the fixed GigaAM bundle:

```bash
uv sync --locked --extra gigaam
.venv/bin/local-transcriber-model --help
.venv/bin/local-transcriber-model install gigaam
```

The installer downloads the pinned Pisar CLI v1.0 ONNX int8 bundle, checks its
exact archive size/SHA-256, safely extracts the five expected files, and uses
the shared CPU loader to validate the staging model before atomic publication.
It does not transcribe audio. The command reports the actual path, profile and
`pinned` verification. Provenance and licenses: [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
The archive is about 214 MB; temporary archive + raw tar + staging need roughly
864 MB, in addition to dependencies and runtime memory.

Without `--model-dir`, installation uses the managed cache:
`${XDG_CACHE_HOME:-~/.cache}/local-transcriber/models/gigaam-v3-e2e-rnnt-int8-v1/`.
It ignores `GIGAAM_MODEL_DIR` and legacy locations when choosing the install target.
For a custom target:

```bash
.venv/bin/local-transcriber-model install gigaam --model-dir /path/to/new-model
```

The target itself must **not exist** (do not pre-create an empty directory).
Parents are created as needed. An existing exact pinned model is revalidated
and succeeds without a download or changes to its files. An empty, invalid,
symlink or compatible-but-unverified target is preserved and rejected: choose
another absent path. There is no update, remove, force, fallback or automatic retry.
Network/checksum/validation/access/lock errors return nonzero with the reason,
without a traceback. Install only into a trusted, user-controlled parent.
A crash can leave a sibling `.<target-name>.install-lock`; there is no automatic
stale-lock removal. Never remove it without confirming no installer owns it.
Cleanup can fail after publication: inspect the reported paths before retrying;
an error does not necessarily mean the target was never published.

For Whisper alone, use `uv sync --locked` without the extra. Whisper's initial
weight download is unchanged; prepare its selected weights before offline use.
`uv run --locked --extra gigaam ...` is a possible setup wrapper, **not a network
ban**: it can synchronize the environment. Dependency setup is separate from
model installation and from offline runtime.

## Offline runtime after setup

Use installed commands directly, with dependencies, model weights and external
tools already available. These commands do not perform environment sync:

```bash
.venv/bin/local-transcriber --help
.venv/bin/local-transcriber recording.m4a --engine gigaam --language ru --device cpu
```

For the custom target installed above, select it explicitly or via the environment:

```bash
.venv/bin/local-transcriber recording.m4a \
  --engine gigaam --gigaam-model-dir /path/to/new-model
GIGAAM_MODEL_DIR=/path/to/new-model .venv/bin/local-transcriber recording.m4a \
  --engine gigaam
```

The positional audio CLI remains unchanged; installation is a **separate command**.
`.venv/bin/python -m local_transcriber recording.m4a ...` is the equivalent
project-Python entry point. GigaAM ASR does not install/download models, import
Whisper/torch, or fall back to Whisper. Help and argument errors for the model
command work without model files or optional runtime dependencies.

Both GigaAM model loading and the model command's dependency preflight set
`ORT_DISABLE_TELEMETRY=1` before their first ONNX Runtime import, overriding any
inherited value (including `0`). This uses ORT's full pre-initialization opt-out:
ORT 1.30 can initialize Microsoft 1DS telemetry during import, so calling
`disable_telemetry_events()` afterward is insufficient. Help and Whisper-only
paths do not import ORT or change this flag. If another library or embedding
application imports ORT first, set `ORT_DISABLE_TELEMETRY=1` **before starting that
process**; this application cannot retroactively undo 1DS initialization. This
policy suppresses the identified ORT telemetry path, not a process-wide network
sandbox or proof that every dependency is network-free.

GigaAM supports only Russian (`--language ru` or omitted) and CPU (`--device auto`
or `cpu`), with the fixed `v3_e2e_rnnt` ONNX int8 profile. Other languages/devices,
`--initial-prompt` and `--prompt-speakers` are rejected. PCM chunks are at most
24 seconds, split near pauses or by hard cuts. Timestamps are approximate chunk
boundaries, not word alignment; there is no diarization. `--speaker-count` is
metadata only. Unit/help checks are not evidence of offline behavior for a real
ASR process tree; that requires the separate real integration gate described below.

Whisper remains the default, with automatic language/device detection:

```bash
.venv/bin/local-transcriber recording.mp4
.venv/bin/local-transcriber recording.m4a \
  --language ru --whisper-model turbo --out-dir out
.venv/bin/local-transcriber interview.m4a \
  --initial-prompt "Discussion about Kubernetes, PostgreSQL, and billing APIs."
```

These Whisper examples are offline only when the selected weights are already
cached. On Apple Silicon, automatic Whisper device selection uses MPS when
available. If an operation is unsupported, enable PyTorch's CPU fallback:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 .venv/bin/local-transcriber recording.m4a
```

### Model paths and trust

ASR searches in this order: explicit `--gigaam-model-dir` → `GIGAAM_MODEL_DIR` →
managed cache → legacy `~/.giga/model`. Paths expand `~` and resolve to the actual
absolute directory. An invalid explicit/env path or a found but damaged cache
stops the run; only an absent cache permits legacy fallback. Installation's
`--model-dir` is a write target, not this ASR search policy.

The loader checks the five files, fixed YAML profile, tokenizer and named ONNX
signatures. `pinned` means all five file hashes match the trusted bundle hashes,
not just a URL/tag or a claim in a manifest. A structurally compatible manual
folder may be used for ASR as `unverified`; different bytes (even edited YAML)
are not an exact pinned bundle. The installer will not overwrite that folder.

`model-manifest.json` records profile, bundle checksum, versions and file sizes/
hashes. Absence of a manifest is allowed; a damaged or inconsistent manifest is
an error, not a reason to downgrade to manual or try a different model. A
self-consistent user manifest does not establish pinned trust. Structural shapes
and tokenizer size do not prove correct semantics or safety: **the validator is
not a sandbox for arbitrary ONNX models**. Keep model files unchanged during use.

### Verified environment and platform limits

Recorded checks on **2026-09-25** passed **124 tests**, Ruff, ty and installed
CLI help on both **macOS 27.0 / arm64** and **Debian bookworm / Linux arm64**,
with Python 3.13.3. The macOS environment used ONNX Runtime 1.30.0, NumPy 2.4.4,
SentencePiece 0.2.2, PyYAML 6.0.3 and ffmpeg/ffprobe 8.1.2.

Separate real-model checks verified a fresh download, pinned-model validation,
native no-replace publication and unchanged idempotent reinstallation on macOS.
After the telemetry fix, same-input reference parity passed all seven cases,
and two macOS CLI runs verified JSON v1, processing/RAW-index mappings, full
audio duration, pinned model identity, both Markdown outputs and preservation
of previous results. Wall time and peak process RSS were measured; these are
bounded observations, not performance guarantees. The fresh download preceded
the telemetry-only fix; model loading, existing-target installation, parity and
ASR were rechecked afterward, without another download.

Two additional real CLI runs on Linux arm64 used the same pinned model in an
isolated local container. Qualified full-process-tree syscall tracing observed
**zero network/socket-FD operations** in both runs, and output checks passed.
This evidence applies to those two runs and that Linux environment, not to all
inputs or platforms. It does **not** establish complete macOS network tracing;
the macOS collector had known coverage limits. A separate real Linux model
download/install was not repeated.

Other platforms (including Windows), missing libc capabilities, or unsupported
filesystems fail closed at publication with no unsafe rename/copy fallback;
download/validation may already have happened by then. Python 3.12+ and available
wheels do not promise all OS/architecture pairs. Unit tests, reference parity
and real integration checks are distinct evidence; none is a universal accuracy,
network-absence or release-readiness guarantee.

## Outputs

Each run gets a new directory under `--out-dir` (default: `out/`):

```text
out/
  recording/
    2026-09-12_14-30-05/
      transcript.md
      transcript_timestamps.md
      transcript.json
```

- `transcript.md` — plain transcript;
- `transcript_timestamps.md` — transcript with timestamps;
- `transcript.json` — engine/model metadata, `raw_segments` as returned by the
  selected engine, and cleaned `merged_segments` with only
  `start`/`end`/`text`.

Whisper raw segments retain their confidence metrics; metrics are not averaged
across merged segments. GigaAM raw segments are approximate timestamped chunks,
recorded with `timestamp_kind: "chunk"`, and remain separate after filtering.

### JSON transcript contract (v1)

New `transcript.json` files add these fields without removing the previous metadata:

- `artifact_type: "transcript"` and `schema_version: 1`;
- `processing`: `min_segment_seconds`, `drop_subtitle_artifacts`,
  `merge_gap_seconds`, and the actual `merge_policy` (`adjacent` for Whisper,
  `none` for GigaAM, even when a merge gap was requested);
- `view_raw_indices`: one list of zero-based `raw_segments` indices per
  `merged_segments` entry. Filtering never renumbers RAW. For example,
  `[[1, 4], [5]]` means the first view segment merges RAW entries 1 and 4,
  and the second comes from entry 5. GigaAM lists always contain one index;
- `duration_seconds` for GigaAM: the complete normalized WAV frame count divided
  by 16000, including trailing silence and recordings with no recognized speech.
  Whisper omits this field when no exact duration is available; the last speech
  timestamp is not a substitute for audio duration.

RAW is the engine result, **not ground truth**. It retains original text and all
engine-specific segment fields, including Whisper tokens and confidence metrics.
Trimming, filtering, and merging create a separate view; they do not edit RAW.
GigaAM timestamps describe approximate chunk boundaries, not word alignment or
exact speech onset/end. A filtered or empty view does not shorten audio duration.

GigaAM emits `model_identity` from the shared loader: profile, actual absolute
`model_dir`, `verification: pinned|unverified`, and the measured per-file SHA-256
mapping. `bundle_sha256` is included only for an exact trusted pinned match.
Existing `gigaam_profile` and `gigaam_model_dir` are retained. Hashes and runtime
objects are reused, not recomputed/reloaded by the CLI. Whisper does not receive
a fictitious ONNX identity. No glossary or corrected artifacts are produced.

Serialization produces UTF-8 JSON bytes once (`ensure_ascii=False`, two-space
indentation, no appended newline); the writer publishes exactly those bytes.
A future source SHA-256 must hash the actual `transcript.json` bytes, not another
JSON dump. RAW contains no self-referential file hash.

Legacy JSON files are not rewritten or silently upgraded. Missing version,
processing, mapping, duration, or model facts must not be guessed for replay:
they require independently proven facts in a separate derived manifest, or a new
RAW from an authorized run. A replay/legacy loader is not implemented yet.

### File safety

The directory uses the input stem and local start time. JSON records `started_at`
with its UTC offset. Runs starting in the same second get suffixes `-2`, `-3`,
etc.; directory creation reserves each name exclusively, including concurrent
runs. Inputs with the same stem share a parent, but never a run directory.
Previous results are not overwritten; `--overwrite` has been removed.

The run directory is created before loading the model. A failed run may leave
an empty or incomplete directory. All three output paths are checked before the
first write, including dangling symlinks, and every file uses exclusive creation
after that check. Files are written individually, not as an atomic transaction:
a race or a real I/O failure can leave partial output, but never overwrites an
existing result.

Subtitle/credit filtering is disabled by default. `--drop-subtitle-artifacts`
enables a keyword heuristic that can also remove legitimate speech mentioning
subtitles or phrases such as “created by”. `--min-segment-seconds` optionally
drops short segments. `--merge-gap-seconds` controls Whisper merging (default:
1.5); GigaAM chunk boundaries are not merged.

For Whisper, explicit `--device cuda` or `--device mps` requires that backend
to be available in PyTorch; otherwise the CLI reports an error before loading
the model. Backend availability does not guarantee that every Whisper operation
is supported.

`--speaker-count` only records expected speaker count in metadata. With
`--prompt-speakers`, it is also included as weak prompt context. Whisper does
not label or separate speakers.

## Development

Internal packages separate inference from model lifecycle:

- `local_transcriber.gigaam`: `audio` handles conversion, WAV reads and chunking;
  `features` computes log-mel features; `engine` owns inference and decoding.
  `GigaAMEngine` remains available from `local_transcriber.gigaam`.
- `local_transcriber.models`: `validation` owns the pinned specification, model
  identity and shared loader; `installer` handles explicit model setup;
  `runtime_policy` disables ORT telemetry before import. Importing the package
  alone does not load its submodules or optional dependencies.
- `cli` and `models_cli` remain separate entry points. `transcript` and `outputs`
  remain shared engine-independent modules.

The shared APIs are independent of argparse and ASR imports:

- `local_transcriber.transcript`: `TranscriptResult`, `RunMetadata`,
  `ProcessingPolicy`, `TranscriptSegment`, `TranscriptView`, `select_segments`,
  and `process_transcript`. Selected segments carry `raw_indices`; use those
  indices to access the exact RAW text (the view text is stripped).
- `local_transcriber.outputs`: `serialize_transcript` returns JSON bytes;
  `serialize_outputs` returns `SerializedOutputs`; `write_outputs(out_dir,
  outputs)` exclusively writes those bytes. `create_run_directory` reserves a
  unique run directory. Serializers receive result/view/metadata, not CLI args.
- `GigaAMEngine.transcribe_result` returns a `TranscriptResult` with full WAV
  duration. The existing `transcribe` method remains a speech-chunk list adapter.
- Existing CLI processing imports remain compatibility re-exports. The legacy
  `cli.write_outputs` adapter requires the supplied segments to match RAW and
  the requested policy; it rejects inconsistent inputs rather than guessing
  provenance or recording a false policy.

RAW and model-identity dictionaries are defensively copied when constructing the
frozen result/metadata containers. Consumers must treat these nested snapshots
as read-only; shared processing and serialization never mutate them.

Install the optional backend dependencies and run the development checks:

```bash
uv sync --locked --extra gigaam
uv run --locked --extra gigaam python -m unittest discover
uv run --locked --extra gigaam ruff check .
uv run --locked --extra gigaam ty check
```

Dependency setup may use the network; `uv run --locked` is not a network ban.
For offline unit tests in an already prepared project environment, use the
installed tools directly (also avoiding an unrelated inherited `VIRTUAL_ENV`):

```bash
env -u VIRTUAL_ENV .venv/bin/python -m unittest discover
env -u VIRTUAL_ENV .venv/bin/ruff check .
env -u VIRTUAL_ENV .venv/bin/ty check
```

Tests use synthetic fixtures and mocked inference, not models or private recordings.
