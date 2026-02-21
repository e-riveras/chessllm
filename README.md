# chessllm

Fine-tuning Llama 3.2 3B for chess position analysis using LoRA on Apple Silicon.

Built to run fully locally on a 16GB M4 MacBook Air using [MLX](https://github.com/ml-explore/mlx-examples/tree/main/llms), Apple's native ML framework — no cloud GPUs needed.

## Model

| | |
|---|---|
| Base model | `mlx-community/Llama-3.2-3B-Instruct-4bit` |
| Parameters | 3 billion |
| Quantization | 4-bit (MLX) |
| Fine-tuning | LoRA (rank 8) |
| Task | Chess position analysis / commentary |

## Project Structure

```
chessllm/
├── data/
│   ├── prepare.py        # converts raw chess data → training JSONL
│   ├── raw/              # place annotated PGN files here (git-ignored)
│   └── processed/        # generated train/valid splits (git-ignored)
├── train/
│   ├── lora.yaml         # LoRA hyperparameters
│   └── train.sh          # training entrypoint
├── inference/
│   └── chat.py           # interactive analysis CLI
├── scripts/
│   └── download_model.py # fetch model from HuggingFace
├── adapters/             # saved LoRA adapters (git-ignored)
└── environment.yml
```

## Setup

```bash
# 1. Create and activate the conda environment
conda env create -f environment.yml
conda activate chessllm

# 2. Download the base model (~2GB)
python scripts/download_model.py
```

## Data Preparation

**From annotated PGN files** (place `.pgn` files in `data/raw/`):
```bash
python data/prepare.py --source pgn
```

**From Lichess dataset** (downloads from HuggingFace):
```bash
python data/prepare.py --source lichess --max-examples 5000
```

Both commands output `data/processed/train.jsonl` and `data/processed/valid.jsonl`.

## Training

```bash
./train/train.sh
```

Override any config option inline:
```bash
./train/train.sh --iters 2000 --batch-size 8
```

Training progress and validation loss are logged to the terminal. Adapters are saved to `adapters/chess-lora/` every 200 steps.

## Inference

**Interactive mode:**
```bash
python inference/chat.py
```

**Single position:**
```bash
python inference/chat.py --fen "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK02R b KQkq - 3 3"
```

**Base model only (no adapter):**
```bash
python inference/chat.py --no-adapter
```

## Memory Usage (estimated, 16GB M4)

| Stage | Approx. VRAM |
|---|---|
| Model load (4-bit) | ~2 GB |
| LoRA training (rank 8, batch 4) | ~8–10 GB |
| Inference with adapter | ~3 GB |

## Roadmap

- [x] LoRA supervised fine-tuning (SFT)
- [ ] DPO fine-tuning (`data/prepare.py --dpo` already outputs pairs)
- [ ] Evaluation against Stockfish annotations
- [ ] Export fused model for Ollama / llama.cpp
