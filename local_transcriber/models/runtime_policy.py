"""Lightweight runtime privacy policy; importing this module loads no engines."""

import os


def disable_ort_telemetry() -> None:
    """Apply the full opt-out before the first application-owned ORT import.

    ORT can initialize 1DS during import, before its Python telemetry API is usable.
    External callers that already imported ORT must opt out before process startup.
    """
    os.environ["ORT_DISABLE_TELEMETRY"] = "1"
