"""
Download the base model from HuggingFace via mlx-lm.

Usage:
    python scripts/download_model.py
    python scripts/download_model.py --model mlx-community/Llama-3.2-1B-Instruct-4bit
"""

import argparse
import subprocess
import sys

DEFAULT_MODEL = "mlx-community/Llama-3.2-3B-Instruct-4bit"


def main():
    parser = argparse.ArgumentParser(description="Download an MLX-quantized model.")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"HuggingFace model ID to download (default: {DEFAULT_MODEL})",
    )
    args = parser.parse_args()

    print(f"Downloading: {args.model}")
    print("This may take a few minutes on first run...\n")

    cmd = [
        sys.executable, "-m", "mlx_lm.convert",
        "--hf-path", args.model,
        "--mlx-path", f"models/{args.model.split('/')[-1]}",
    ]

    # mlx-community models are already converted — just snapshot download
    from huggingface_hub import snapshot_download
    local_dir = snapshot_download(
        repo_id=args.model,
        local_dir=f"models/{args.model.split('/')[-1]}",
        ignore_patterns=["*.pt", "*.bin"],  # skip PyTorch weights if present
    )
    print(f"\nModel saved to: {local_dir}")


if __name__ == "__main__":
    main()
