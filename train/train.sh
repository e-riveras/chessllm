#!/usr/bin/env bash
# Fine-tune Llama 3.2 3B with LoRA using mlx-lm.
#
# Usage:
#   ./train/train.sh                    # use defaults from lora.yaml
#   ./train/train.sh --iters 2000       # override any yaml option

set -euo pipefail

CONFIG="train/lora.yaml"

echo "==> Starting LoRA fine-tuning"
echo "    Config: $CONFIG"
echo "    Extra args: $*"
echo ""

python -m mlx_lm.lora \
  --config "$CONFIG" \
  "$@"

echo ""
echo "==> Training complete. Adapters saved to: adapters/chess-lora"
echo ""
echo "To run inference with the adapter:"
echo "  python inference/chat.py"
