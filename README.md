# local-transcriber

Private, local audio and video transcription powered by
[`openai-whisper`](https://github.com/openai/whisper). The recording stays on
your machine; no hosted model or diarization service is used.

The CLI accepts Russian, English, and other languages supported by Whisper. It
writes a plain Markdown transcript, a timestamped version, and structured JSON.

## Requirements

- Python 3.12+
- [`uv`](https://docs.astral.sh/uv/)
- `ffmpeg`

## Usage

Install dependencies and show the available options:

```bash
uv sync
uv run local-transcriber --help
```

Transcribe a recording with automatic language and device detection:

```bash
uv run local-transcriber recording.mp4
```

Select a language, model, or output directory:

```bash
uv run local-transcriber recording.m4a \
  --language ru \
  --whisper-model turbo \
  --out-dir out
```

Give Whisper useful vocabulary or proper names:

```bash
uv run local-transcriber interview.m4a \
  --initial-prompt "Discussion about Kubernetes, PostgreSQL, and billing APIs."
```

On Apple Silicon, automatic device selection uses MPS when available. If an
operation is unsupported, enable PyTorch's CPU fallback:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 uv run local-transcriber recording.m4a
```

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
- `transcript.json` — metadata, `raw_segments` as returned by Whisper (including
  confidence metrics), and cleaned `merged_segments` with only `start`/`end`/`text`.

The JSON keys replace the previous `segments` field. Raw segments are retained
before filtering and merging; metrics are not averaged across merged segments.

The directory uses the input stem and local start time. JSON records `started_at`
with its UTC offset. Runs starting in the same second get suffixes `-2`, `-3`,
etc.; directory creation reserves each name exclusively, including concurrent
runs. Inputs with the same stem share a parent, but never a run directory.
Previous results are not overwritten; `--overwrite` has been removed.

The run directory is created before loading the model. A failed run may leave
an empty or incomplete directory; files are written individually, not atomically.

Subtitle/credit filtering is disabled by default. `--drop-subtitle-artifacts`
enables a keyword heuristic that can also remove legitimate speech mentioning
subtitles or phrases such as “created by”. `--min-segment-seconds` optionally
drops short segments; `--merge-gap-seconds` controls merging (default: 1.5).

Explicit `--device cuda` or `--device mps` requires that backend to be available
in PyTorch; otherwise the CLI reports an error before loading the model. Backend
availability does not guarantee that every Whisper operation is supported.

`--speaker-count` only records expected speaker count in metadata. With
`--prompt-speakers`, it is also included as weak prompt context. Whisper does
not label or separate speakers.

## Development

```bash
uv run python -m unittest discover
uv run ruff check .
uv run ty check
```
