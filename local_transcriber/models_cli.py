"""Explicit model setup, separate from the positional transcription CLI."""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path

from local_transcriber.models.runtime_policy import disable_ort_telemetry


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="local-transcriber-model",
        description="Explicit online model setup; transcription never installs GigaAM.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    install = commands.add_parser("install", help="Install the fixed, pinned bundle.")
    install.add_argument("model", choices=("gigaam",))
    install.add_argument(
        "--model-dir",
        type=Path,
        default=None,
        metavar="TARGET",
        help=(
            "Installation target (must not exist unless already exact pinned). "
            "Default: managed cache; ignores GIGAAM_MODEL_DIR and legacy paths."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
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
