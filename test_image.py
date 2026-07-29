#!/usr/bin/env python3
"""
test_image.py - Command-line tester for the IRA Flux image model.

Exercises the EXACT code path the bot uses (ai_router.generate_image_with_meta)
so you can verify the Hugging Face image pipeline end-to-end from the terminal,
without needing Telegram.

Usage:
    # Generate one image with a custom prompt (saved to demo_output/):
    python3 test_image.py "a neon koi fish swimming through the stars"

    # Use a built-in example prompt:
    python3 test_image.py --example

    # Pick where to save it and which extension to assume:
    python3 test_image.py "a samurai cat" -o result.png

Environment:
    Requires an HF token: HF_TOKEN_1, HF_TOKEN_2, or HF_TOKEN.
    (Copy .env.example -> .env, or export them in your shell.)

Exit codes:
    0  success  (image saved)
    1  config error (no token / missing deps)
    2  provider error (every HF endpoint failed)
"""

import argparse
import os
import sys
import time
from pathlib import Path

# Allow running from the repo root without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # python-dotenv is optional; real deployments use real env vars
    pass

EXAMPLE_PROMPT = (
    "a majestic snow leopard resting on a cliff at golden hour, "
    "ultra detailed, cinematic lighting, 8k"
)

# PNG magic bytes -> a quick sniff to pick a sensible default extension
_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
_JPEG_MAGIC = b"\xff\xd8\xff"
_WEBP_MAGIC = b"RIFF"


def _sniff_ext(image_bytes: bytes, fallback: str = "png") -> str:
    """Guess a file extension from the first bytes of an image."""
    if image_bytes.startswith(_PNG_MAGIC):
        return "png"
    if image_bytes.startswith(_JPEG_MAGIC):
        return "jpg"
    if image_bytes.startswith(_WEBP_MAGIC):
        return "webp"
    return fallback


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Test the IRA Flux image model from the command line.",
    )
    parser.add_argument(
        "prompt",
        nargs="?",
        help='Image prompt, e.g. "a red fox in snow, cinematic". '
             "Omit to use the built-in example.",
    )
    parser.add_argument(
        "--example", action="store_true",
        help="Use the built-in example prompt.",
    )
    parser.add_argument(
        "-o", "--output", default=None,
        help="Output file path. Defaults to demo_output/<timestamp>.<ext>.",
    )
    args = parser.parse_args()

    # Resolve prompt
    if args.example and not args.prompt:
        prompt = EXAMPLE_PROMPT
    elif args.prompt:
        prompt = args.prompt
    else:
        prompt = EXAMPLE_PROMPT
        print("ℹ️  No prompt given — using built-in example. "
              "Pass --help for options.\n")

    # Import the router (deferred so --help works even if deps are missing)
    try:
        from ai_router import generate_image_with_meta, IRAProviderError
    except ImportError as exc:
        print(f"❌ Could not import the image router: {exc}", file=sys.stderr)
        print("   Run this script from the repo root and install requirements.txt.",
              file=sys.stderr)
        return 1

    # Config check up front
    token_set = bool(
        os.getenv("HF_TOKEN_1") or os.getenv("HF_TOKEN_2") or os.getenv("HF_TOKEN")
    )
    if not token_set:
        print("❌ No Hugging Face token found in the environment.", file=sys.stderr)
        print("   Set HF_TOKEN_1 (or HF_TOKEN_2 / HF_TOKEN) before running.",
              file=sys.stderr)
        return 1

    print("🎨 Generating image…")
    print(f"   prompt: {prompt}\n")

    start = time.monotonic()
    try:
        meta = generate_image_with_meta(prompt)
    except ValueError as exc:
        print(f"❌ Config error: {exc}", file=sys.stderr)
        return 1
    except IRAProviderError as exc:
        print(f"❌ Every HF endpoint failed: {exc}", file=sys.stderr)
        print("   (FLUX.1-dev may be cold-starting on the free tier; "
              "retry in a minute.)", file=sys.stderr)
        return 2
    total = time.monotonic() - start

    image_bytes = meta["image"]
    ext = _sniff_ext(image_bytes, fallback="png")

    # Decide output path
    if args.output:
        out_path = Path(args.output)
        # Honour an explicit extension, otherwise add the sniffed one
        if out_path.suffix:
            ext = out_path.suffix.lstrip(".")
        else:
            out_path = out_path.with_suffix(f".{ext}")
    else:
        out_dir = Path("demo_output")
        out_dir.mkdir(exist_ok=True)
        ts = time.strftime("%Y%m%d-%H%M%S")
        out_path = out_dir / f"image-{ts}.{ext}"

    out_path = out_path.resolve()

    # Ensure the parent directory exists (for both default and -o paths)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(image_bytes)

    # Report
    size_kb = meta["size_bytes"] / 1024
    print("✅ Success!")
    print(f"   saved to  : {out_path}")
    print(f"   model     : {meta['model']}")
    print(f"   endpoint  : {meta['endpoint']}")
    print(f"   content   : {meta['content_type']} ({ext.upper()})")
    print(f"   size      : {size_kb:.1f} KB ({meta['size_bytes']:,} bytes)")
    print(f"   model time: {meta['elapsed_ms']} ms")
    print(f"   total time: {total*1000:.0f} ms (incl. download + decode)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
