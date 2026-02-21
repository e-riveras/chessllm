"""
Prepare chess analysis data for fine-tuning.

Converts raw sources into the JSONL format expected by mlx-lm:
  {"messages": [{"role": "user", ...}, {"role": "assistant", ...}]}

Supported input formats:
  --source pgn     : annotated PGN files with commentary (data/raw/*.pgn)
  --source lichess : Lichess puzzle + analysis dataset from HuggingFace

DPO output (--dpo) produces pairs:
  {"prompt": [...], "chosen": [...], "rejected": [...]}

Usage:
    python data/prepare.py --source lichess
    python data/prepare.py --source pgn --pgn-dir data/raw/
    python data/prepare.py --source lichess --dpo
"""

import argparse
import json
import random
import re
from pathlib import Path

import chess
import chess.pgn
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a chess expert. When given a chess position, you provide clear, "
    "accurate analysis suitable for intermediate to advanced players."
)


def make_analysis_prompt(fen: str, context: str = "") -> str:
    parts = [f"Position (FEN): {fen}"]
    if context:
        parts.append(f"Context: {context}")
    parts.append("Analyze this position.")
    return "\n".join(parts)


def make_messages(user_content: str, assistant_content: str) -> dict:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ]
    }


# ---------------------------------------------------------------------------
# PGN source
# ---------------------------------------------------------------------------

def iter_pgn_examples(pgn_dir: Path):
    """
    Extract (FEN, commentary) pairs from annotated PGN files.
    Skips moves without comments.
    """
    for pgn_path in sorted(pgn_dir.glob("*.pgn")):
        print(f"  Processing {pgn_path.name}")
        with open(pgn_path, encoding="utf-8", errors="ignore") as f:
            while True:
                game = chess.pgn.read_game(f)
                if game is None:
                    break

                board = game.board()
                node = game

                while node.variations:
                    node = node.variation(0)
                    board.push(node.move)
                    comment = node.comment.strip()

                    # skip empty or very short comments
                    if len(comment) < 30:
                        continue

                    fen = board.fen()
                    move_san = node.san() if hasattr(node, "san") else ""
                    context = f"Last move played: {move_san}" if move_san else ""

                    yield make_messages(
                        make_analysis_prompt(fen, context),
                        comment,
                    )


# ---------------------------------------------------------------------------
# Lichess source (HuggingFace dataset)
# ---------------------------------------------------------------------------

def iter_lichess_examples(max_examples: int = 5000):
    """
    Load the 'lichess_elite' or similar HF dataset and build analysis pairs.
    Falls back to a small synthetic demo set if the dataset isn't available.
    """
    try:
        from datasets import load_dataset
        # niklasf/python-chess has no direct commentary dataset on HF yet;
        # use 'adamkarvonen/chess_games' as a stand-in for PGN text.
        ds = load_dataset("adamkarvonen/chess_games", split="train", streaming=True)
        count = 0
        for row in tqdm(ds, total=max_examples, desc="Lichess"):
            if count >= max_examples:
                break
            pgn_text = row.get("transcript", "")
            if not pgn_text:
                continue
            import io
            game = chess.pgn.read_game(io.StringIO(pgn_text))
            if game is None:
                continue
            # Build a simple material + turn analysis as placeholder commentary
            board = game.board()
            moves = list(game.mainline_moves())
            if len(moves) < 10:
                continue
            # Jump to a random middle position
            cutoff = random.randint(10, min(len(moves), 40))
            for m in moves[:cutoff]:
                board.push(m)
            fen = board.fen()
            analysis = _simple_material_summary(board)
            yield make_messages(make_analysis_prompt(fen), analysis)
            count += 1
    except Exception as e:
        print(f"Warning: could not load HF dataset ({e}). Using synthetic demo data.")
        yield from _synthetic_examples()


def _simple_material_summary(board: chess.Board) -> str:
    piece_values = {
        chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
        chess.ROOK: 5, chess.QUEEN: 9,
    }
    white_mat = sum(
        len(board.pieces(pt, chess.WHITE)) * v for pt, v in piece_values.items()
    )
    black_mat = sum(
        len(board.pieces(pt, chess.BLACK)) * v for pt, v in piece_values.items()
    )
    turn = "White" if board.turn == chess.WHITE else "Black"
    diff = white_mat - black_mat
    if diff > 0:
        material = f"White has a material advantage of {diff} points."
    elif diff < 0:
        material = f"Black has a material advantage of {abs(diff)} points."
    else:
        material = "Material is equal."

    in_check = " The king is in check." if board.is_check() else ""
    return f"{turn} to move. {material}{in_check}"


def _synthetic_examples():
    """A handful of hand-crafted examples for smoke testing."""
    examples = [
        (
            "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3",
            "This is the Italian Game after 1.e4 e5 2.Nf3 Nc6 3.Bc4. "
            "White places the bishop on the active c4 diagonal, eyeing the f7 pawn. "
            "Black's most principled responses are 3...Bc5 (Giuoco Piano) or 3...Nf6 (Two Knights).",
        ),
        (
            "rnbqkb1r/pp2pppp/2p2n2/3p4/2PP4/2N5/PP2PPPP/R1BQKBNR w KQkq d6 0 4",
            "The Slav Defense. Black has chosen a solid pawn structure with pawns on c6 and d5. "
            "The c8 bishop remains unblocked unlike in the Queen's Gambit Declined. "
            "White should consider 4.Nf3 or 4.e3 to complete development.",
        ),
    ]
    for fen, commentary in examples:
        yield make_messages(make_analysis_prompt(fen), commentary)


# ---------------------------------------------------------------------------
# DPO pair generation
# ---------------------------------------------------------------------------

def to_dpo_pair(example: dict) -> dict:
    """
    Wraps a good analysis example into a DPO pair by generating a weak
    'rejected' response. In practice, replace rejected with actual weak
    model outputs or human-ranked data.
    """
    messages = example["messages"]
    chosen = [m for m in messages if m["role"] == "assistant"]
    assert chosen

    rejected_content = "I'm not sure about this position. It looks complicated."
    rejected = [{"role": "assistant", "content": rejected_content}]

    prompt = [m for m in messages if m["role"] != "assistant"]
    return {
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def write_splits(examples: list, out_dir: Path, dpo: bool):
    random.shuffle(examples)
    n_valid = max(1, int(len(examples) * 0.05))
    train = examples[n_valid:]
    valid = examples[:n_valid]

    out_dir.mkdir(parents=True, exist_ok=True)

    for split_name, split_data in [("train", train), ("valid", valid)]:
        out_path = out_dir / f"{split_name}.jsonl"
        with open(out_path, "w") as f:
            for ex in split_data:
                row = to_dpo_pair(ex) if dpo else ex
                f.write(json.dumps(row) + "\n")
        print(f"Wrote {len(split_data):,} examples → {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["pgn", "lichess"], default="lichess")
    parser.add_argument("--pgn-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--max-examples", type=int, default=5000)
    parser.add_argument("--dpo", action="store_true", help="Output DPO pairs instead of SFT")
    args = parser.parse_args()

    print(f"Source: {args.source} | DPO: {args.dpo}")

    if args.source == "pgn":
        examples = list(tqdm(iter_pgn_examples(args.pgn_dir), desc="PGN"))
    else:
        examples = list(iter_lichess_examples(args.max_examples))

    if not examples:
        print("No examples found. Add annotated PGN files to data/raw/ or check your HF dataset.")
        return

    print(f"\nTotal examples: {len(examples):,}")
    write_splits(examples, args.out_dir, dpo=args.dpo)


if __name__ == "__main__":
    main()
