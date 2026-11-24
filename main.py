"""Face Recognition Service - command line entry point.

Usage
-----
    python main.py              Start the interactive application.
    python main.py --check      Verify the installation (no camera needed).
    python main.py --check --camera
                                Also test the webcam.
    python main.py --clear-cache
                                Delete cached face embeddings.
"""

from __future__ import annotations

import argparse
import sys
from typing import Optional, Sequence

from core import __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="face-recognition",
        description="Register people with a webcam and recognize them later.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="run installation and model checks, then exit",
    )
    parser.add_argument(
        "--camera",
        action="store_true",
        help="with --check, also verify that the webcam can be opened",
    )
    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="delete cached face embeddings and exit",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    return parser


def run_menu() -> None:
    """Run the interactive menu until the user chooses Exit."""

    from core.recognition import check_face
    from core.registration import register_face

    while True:
        print("\n==============================")
        print("     FACE RECOGNITION SYSTEM")
        print("==============================")
        print("1. Register New Person")
        print("2. Recognize Person")
        print("3. Exit")

        try:
            choice = input("\nChoose an option: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            return

        if choice == "1":
            register_face()
        elif choice == "2":
            check_face()
        elif choice == "3":
            print("Goodbye!")
            return
        else:
            print("Invalid option. Try again.")


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    # UTF-8 must be configured before DeepFace prints emoji log messages.
    from core.preflight import ensure_utf8

    ensure_utf8()

    if args.check:
        from core.preflight import run_self_test

        return run_self_test(check_camera_flag=args.camera)

    if args.clear_cache:
        from core.cache import clear_cache

        if clear_cache():
            print("Embedding cache cleared.")
        else:
            print("No embedding cache found.")
        return 0

    if args.camera:
        print("Note: --camera is only used together with --check.")

    from core.preflight import preflight_for_app

    if not preflight_for_app():
        return 1

    try:
        run_menu()
    except KeyboardInterrupt:
        print("\nGoodbye!")

    return 0


if __name__ == "__main__":
    sys.exit(main())
