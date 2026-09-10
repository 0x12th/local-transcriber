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

By default, files are written to `out/`:

- `transcript.md` — plain transcript;
- `transcript_timestamps.md` — transcript with timestamps;
- `transcript_data.json` — metadata and `start`/`end`/`text` segments.

`--speaker-count` only records expected speaker count in metadata. With
`--prompt-speakers`, it is also included as weak prompt context. Whisper does
not label or separate speakers.

## Development

```bash
uv run python -m unittest discover
uv run ruff check .
uv run ty check
```
