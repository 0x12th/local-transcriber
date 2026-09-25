"""Local GigaAM v3 ONNX backend adapted from the pinned Giga Pisar core.

Source and license details are recorded in THIRD_PARTY_NOTICES.md.
"""

from local_transcriber.gigaam.audio import (
    MAX_CHUNK,
    MAX_CHUNK_SAMPLES,
    MIN_PAUSE_OFFSET_SAMPLES,
    SAMPLE_RATE,
    SILENCE_DB,
    SILENCE_MIN,
    chunk_bounds,
    chunk_sample_bounds,
    find_silences,
)
from local_transcriber.gigaam.engine import (
    MAX_SYMBOLS_PER_FRAME,
    MODEL_NAME,
    GigaAMEngine,
    GigaAMSegment,
    find_model_dir,
)
from local_transcriber.gigaam.features import Features

__all__ = [
    "MAX_CHUNK",
    "MAX_CHUNK_SAMPLES",
    "MAX_SYMBOLS_PER_FRAME",
    "MIN_PAUSE_OFFSET_SAMPLES",
    "MODEL_NAME",
    "SAMPLE_RATE",
    "SILENCE_DB",
    "SILENCE_MIN",
    "Features",
    "GigaAMEngine",
    "GigaAMSegment",
    "chunk_bounds",
    "chunk_sample_bounds",
    "find_model_dir",
    "find_silences",
]
