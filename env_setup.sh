#!/bin/bash
set -euo pipefail

uv sync --extra lint

# hpsv2 ships without bpe_simple_vocab_16e6.txt.gz; fetch it into the open_clip
# package directory so the tokenizer can load it at runtime.
HPSV2_OPEN_CLIP_DIR="$(uv run python -c 'import os, hpsv2; print(os.path.join(os.path.dirname(hpsv2.__file__), "src", "open_clip"))')"
BPE_VOCAB_PATH="${HPSV2_OPEN_CLIP_DIR}/bpe_simple_vocab_16e6.txt.gz"
BPE_VOCAB_URL="https://huggingface.co/OpenGVLab/ViCLIP-B-16-hf/resolve/main/bpe_simple_vocab_16e6.txt.gz"

if [ ! -f "${BPE_VOCAB_PATH}" ]; then
    echo "Downloading bpe_simple_vocab_16e6.txt.gz to ${BPE_VOCAB_PATH}"
    curl -fL --retry 3 -o "${BPE_VOCAB_PATH}" "${BPE_VOCAB_URL}"
else
    echo "bpe_simple_vocab_16e6.txt.gz already present at ${BPE_VOCAB_PATH}"
fi
