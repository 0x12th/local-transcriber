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

## Results

Each run creates a separate directory; previous results are not overwritten:

```text
out/<recording>/<run>/
  transcript.md              # Plain transcript
  transcript_timestamps.md   # Transcript with timestamps
  transcript.json            # Metadata, original segments and processed view
```

The JSON preserves original engine output separately from filtering and merging.
GigaAM timestamps mark approximate chunks of up to 24 seconds, not individual
words. Neither engine identifies or labels speakers.

## Useful options

```bash
# Select a Whisper model and language
just transcribe "recording.m4a" --whisper-model small --language ru

# Choose the output directory
just gigaam "recording.m4a" --out-dir "transcripts"

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
