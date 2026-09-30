# local-transcriber

Turn audio or video files into a local transcript: plain text, timestamps and JSON.
Uses **Whisper** for multilingual transcription or **GigaAM** for Russian on CPU.
No hosted transcription service is required.

## Quick start

Requires Python 3.12+, [uv](https://docs.astral.sh/uv/),
[just](https://just.systems/), and `ffmpeg` / `ffprobe` on `PATH`.
Tested on macOS and Linux arm64.

On macOS, install the tools with Homebrew:

```bash
brew install uv just ffmpeg
```

From the project root, install dependencies:

```bash
just dev
```

**Whisper** — the default engine, with automatic language and device selection.
Its first run downloads the selected model weights.

```bash
just transcribe "recording.m4a"
```

**GigaAM** — Russian-only, CPU-only. Install its model once (~214 MB):

```bash
just install-gigaam
just gigaam "recording.m4a"
```

Both engines accept audio and video formats supported by ffmpeg.
Whisper and torch are installed even if you only use GigaAM.

**Speaker-labeled transcript (opt-in, macOS Apple Silicon)** — install Nemotron
once, then enable diarization on the existing Whisper command:

```bash
# Requires Xcode Command Line Tools, Git, CMake >= 3.26, Ninja,
# SentencePiece and Abseil. On macOS: brew install cmake ninja sentencepiece abseil
just install-nemotron
just transcribe "recording.m4a" --diarize
```

`install-nemotron` downloads NVIDIA's pinned GGUF (~107 MB), checks its SHA-256,
checks out a pinned NeMo-Speech.cpp revision and builds its Metal runtime. It
verifies a real inference before publishing to the managed cache. This operation
needs network access and build tools; a fresh download and build have not been
tested on a second machine. The released NeMo-Speech.cpp v0.1.0 is incompatible
with this V3 model. Re-running installation revalidates an existing install
without replacing it. `transcribe --diarize` uses only the installed runtime;
it does not download or install Nemotron. Whisper's normal first-run checkpoint
download still applies. `--diarize` is not supported with GigaAM.

## Results

Each run creates a separate directory; previous results are not overwritten:

```text
out/<recording>/<run>/
  transcript.md              # Plain transcript
  transcript_timestamps.md   # Transcript with timestamps
  transcript.json            # Metadata and compact segments
  transcript_speakers.md     # Only with --diarize: speaker-labeled view
```

`transcript.json` uses schema v2 for Whisper, GigaAM and `--diarize`: an object
with run metadata (`input`, `started_at`, `engine`, model/language/device settings,
etc.) and `segments`. Without `--diarize`, segments contain only `start`, `end`
and `text`; Whisper follows ordinary filtering and merging, while GigaAM keeps
its approximate chunks of up to 24 seconds. With `--diarize`, Whisper keeps the
plain Markdown transcripts, adds `transcript_speakers.md`, and emits postprocessed
speaker turns with an additional `speaker` field. Speaker numbers start at 1 in
order of first assigned appearance; `null` means diarization could not assign
that segment.
`diarization_status` is `success` or `failed` for diarized runs only.

Add `--debug` to any transcription command to also include the previous detailed
fields in the same JSON: `raw_segments` (including engine-specific metrics),
`merged_segments`, `processing`, and `view_raw_indices`. These debug indices
refer to the ordinary ASR view, not speaker turns. Existing run files are not
rewritten; readers of previous schema v1 objects or diarized JSON arrays must
handle both formats if they load old recordings.

After diarization, deterministic cleanup bridges short, continuous same-speaker
gaps (`A → null(s) → A`) and very short in-sentence label flips (`A → B → A`),
then joins adjacent continuous turns of the same speaker. It keeps ambiguous
overlaps, common English and Russian short replies (e.g. `yeah` / `да`) and
speaker-change boundaries separate; text-case, punctuation and timing heuristics
cannot replace checking uncertain passages against the audio. Labels identify
voices *within one recording*, not people. If diarization fails after ASR, the
run retains the normal Markdown transcript, marks the speaker Markdown and JSON
status as failed, writes `null` speakers and exits nonzero.

## Useful options

```bash
# Select a Whisper model and language
just transcribe "recording.m4a" --whisper-model small --language ru

# Choose the output directory
just gigaam "recording.m4a" --out-dir "transcripts"

# Include RAW engine data and processing provenance in transcript.json
just transcribe "recording.m4a" --diarize --debug

# See all CLI options
just transcribe --help
```

For a custom GigaAM model location:

```bash
just install-gigaam --model-dir "models/gigaam"
just gigaam "recording.m4a" --gigaam-model-dir "models/gigaam"
```

Use an absent target for a new installation; do not pre-create an empty folder.
Reinstalling the exact pinned model revalidates it without downloading or replacing
files. Existing different files are not overwritten. Without a custom path, the
installer uses the managed model cache.

## Privacy and limitations

- Transcription runs locally. Dependency setup and model downloads need internet;
  prepare weights before going offline. The run commands do not sync dependencies.
- **ONNX Runtime telemetry is disabled by default**, before GigaAM imports it.
  If embedding the Python API alongside another ORT user, set
  `ORT_DISABLE_TELEMETRY=1` before that process starts. This is not a network sandbox.
- Recognition can miss or mishear words; review important passages against the audio.
- On Apple Silicon, if Whisper encounters an unsupported MPS operation, use
  `PYTORCH_ENABLE_MPS_FALLBACK=1 just transcribe "recording.m4a"` or `--device cpu`.

## Development

```bash
just check                       # Lint, types and tests
just test tests.test_gigaam -v    # Run selected tests
just build                       # Build wheel and source distribution
just                             # List all commands
```

Tests use synthetic fixtures, not private recordings or model downloads.
`just dev --offline` and `just build --offline` work with cached dependencies.

Third-party code and model licenses: [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
