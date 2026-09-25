set shell := ["sh", "-eu", "-c"]
set positional-arguments

# List available commands.
default:
    @just --list

# Install locked dependencies, including GigaAM and development tools (may use network).
dev *args:
    env -u VIRTUAL_ENV uv sync --locked --extra gigaam --group dev "$@"

# Run all tests, or pass unittest arguments (e.g. tests.test_gigaam -v).
test *args="discover":
    env -u VIRTUAL_ENV .venv/bin/python -m unittest "$@"

# Check Python style without modifying files.
lint:
    env -u VIRTUAL_ENV .venv/bin/ruff check .

# Check types using the project environment explicitly.
typecheck:
    env -u VIRTUAL_ENV .venv/bin/ty check --python .venv/bin/python

# Run lint, type checking and the test suite without synchronizing dependencies.
check: lint typecheck test

# Build wheel and source distribution (may use network for build dependencies).
build *args:
    env -u VIRTUAL_ENV uv build "$@"

# Explicitly install the pinned GigaAM model (may use network).
install-gigaam *args:
    env -u VIRTUAL_ENV .venv/bin/local-transcriber-model install gigaam "$@"

# Run the transcription CLI; Whisper remains the default engine.
transcribe +args:
    env -u VIRTUAL_ENV .venv/bin/local-transcriber "$@"

# Transcribe Russian audio with GigaAM on CPU, without environment/model setup.
gigaam +args:
    env -u VIRTUAL_ENV .venv/bin/local-transcriber --engine gigaam --language ru --device cpu "$@"
