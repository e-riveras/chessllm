"""
Interactive chess position analysis using the fine-tuned model.

Usage:
    # With LoRA adapter (after training):
    python inference/chat.py

    # Base model only (no adapter):
    python inference/chat.py --no-adapter

    # One-shot from CLI:
    python inference/chat.py --fen "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3"
"""

import argparse
import sys

import chess
import mlx.core as mx
from mlx_lm import load, generate

SYSTEM_PROMPT = (
    "You are a chess expert. When given a chess position, you provide clear, "
    "accurate analysis suitable for intermediate to advanced players."
)

MODEL_PATH = "models/Llama-3.2-3B-Instruct-4bit"
ADAPTER_PATH = "adapters/chess-lora"


def build_prompt(model, tokenizer, fen: str, context: str = "") -> str:
    user_parts = [f"Position (FEN): {fen}"]
    if context:
        user_parts.append(f"Context: {context}")
    user_parts.append("Analyze this position.")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "\n".join(user_parts)},
    ]
    # apply_chat_template adds the correct instruct tokens for Llama 3
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    return prompt


def validate_fen(fen: str) -> bool:
    try:
        chess.Board(fen)
        return True
    except ValueError:
        return False


def analyze(model, tokenizer, fen: str, context: str = "", max_tokens: int = 512):
    prompt = build_prompt(model, tokenizer, fen, context)
    response = generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=max_tokens,
        verbose=False,
    )
    return response


def interactive_loop(model, tokenizer):
    print("Chess Analysis — type 'quit' to exit\n")
    while True:
        fen = input("FEN: ").strip()
        if fen.lower() in ("quit", "exit", "q"):
            break
        if not validate_fen(fen):
            print("Invalid FEN string. Try again.\n")
            continue
        context = input("Context (optional, press Enter to skip): ").strip()
        print("\nAnalyzing...\n")
        result = analyze(model, tokenizer, fen, context)
        print(f"Analysis:\n{result}\n")
        print("-" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL_PATH)
    parser.add_argument("--adapter", default=ADAPTER_PATH)
    parser.add_argument("--no-adapter", action="store_true")
    parser.add_argument("--fen", help="Analyze a single FEN and exit")
    parser.add_argument("--context", default="", help="Optional move/game context")
    parser.add_argument("--max-tokens", type=int, default=512)
    args = parser.parse_args()

    adapter = None if args.no_adapter else args.adapter
    print(f"Loading model: {args.model}")
    if adapter:
        print(f"Loading adapter: {adapter}")

    model, tokenizer = load(args.model, adapter_path=adapter)

    if args.fen:
        if not validate_fen(args.fen):
            print("Error: invalid FEN string.")
            sys.exit(1)
        result = analyze(model, tokenizer, args.fen, args.context, args.max_tokens)
        print(result)
    else:
        interactive_loop(model, tokenizer)


if __name__ == "__main__":
    main()
