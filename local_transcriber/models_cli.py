"""Explicit model setup, separate from the positional transcription CLI."""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path

from local_transcriber.models.runtime_policy import disable_ort_telemetry


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="local-transcriber-model",
        description="Explicit online model setup; transcription never installs models.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    install = commands.add_parser("install", help="Install the fixed, pinned bundle.")
    install.add_argument("model", choices=("gigaam", "nemotron"))
    install.add_argument(
        "--model-dir",
        type=Path,
        default=None,
        metavar="TARGET",
        help=(
            "Installation target (must not exist unless already exact pinned). "
            "Default: managed cache. Transcription uses the default Nemotron "
            "location; GigaAM ignores legacy model paths."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.model == "nemotron":
        from local_transcriber.models.nemotron import (
            NemotronInstallError,
            install_nemotron,
        )

        try:
            model, executable = install_nemotron(args.model_dir)
        except NemotronInstallError as error:
            parser.exit(1, f"local-transcriber-model: {error}\n")
        print(f"Nemotron ready: {model.parent}\nModel: {model}\nRuntime: {executable}")
        return

    try:
        # Fail before download if the validation runtime is missing or broken.
        # Importing dependencies creates no model sessions; the installer owns loading.
        disable_ort_telemetry()
        for module in ("numpy", "onnxruntime", "sentencepiece", "yaml"):
            importlib.import_module(module)
        from local_transcriber.models import installer as model_installer
    except (ImportError, OSError) as error:
        parser.exit(
            1,
            "local-transcriber-model: GigaAM dependencies could not be imported: "
            f"{error}. Run `uv sync --locked --extra gigaam` in the project "
            "(dependency setup may use the network).\n",
        )

    try:
        result = model_installer.install_model(args.model_dir)
    except model_installer.ModelInstallError as error:
        parser.exit(1, f"local-transcriber-model: {error}\n")

    identity = result.metadata.model_identity()
    status = "Already installed" if result.already_installed else "Installed"
    print(
        f"{status}: {result.model_dir}\n"
        f"Profile: {identity['profile']}; verification: {identity['verification']}"
    )


if __name__ == "__main__":
    main()
