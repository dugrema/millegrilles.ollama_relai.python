#!/bin/env bash

MODELS="/mnt/tmphold/tas/models"
IMG="registry.millegrilles.com/ggml-org/llama.cpp:full-cuda_6767"
#LOGGING_PARAMS="--log-verbosity -1"
LOGGING_PARAMS="--log-verbosity 0"
PARAMS="--n-gpu-layers 99 --ubatch-size 128 --batch-size 1024 --ctx-size 4096 --n-predict 2048"

# ======================
# OpenAI GPT-OSS
# ======================
MODEL="gguf/gpt-oss-20b-UD-Q6_K_XL.gguf"
PARAMS="-a chat --swa-checkpoints 6 -fa 1 -t 4 --n-cpu-moe 16 --ubatch-size 512 --batch-size 8192 --ctx-size 32768 --temp 1.0 --top-p 1.0 --jinja"
# PARAMS="-a chat --swa-checkpoints 4 -fa 1 -t 4 --n-cpu-moe 15 --ubatch-size 512 --ctx-size 16384 --temp 1.0 --top-p 1.0 --jinja"

# ======================
# Gemma 3
# ======================
# MODEL="gguf/gemma-3-4b-it-UD-Q4_K_XL.gguf"
# PARAMS="--n-gpu-layers 99 --mmproj /models/gguf/gemma-3-4b-mmproj-BF16.gguf --ubatch-size 256 --batch-size 8192 --ctx-size 32768 --n-predict 8192 --temp 1.0 --top-k 64 --top-p 0.95 --min-p 0.0 --jinja"
# MODEL="gguf/gemma-3-12b-it-UD-Q2_K_XL.gguf"
# PARAMS="--n-gpu-layers 99 --ubatch-size 48 --batch-size 10240 --ctx-size 10240 --n-predict 4096 --temp 1.0 --top-k 64 --top-p 0.95 --min-p 0.0 --jinja"

# ======================
# Qwen3
# ======================
# MODEL="gguf/Qwen3-4B-Thinking-2507-UD-Q6_K_XL.gguf"
# PARAMS="--n-gpu-layers 99 --ctx-size 12288 --n-predict 8192 --ubatch-size 128 --batch-size 1024 --temp 0.6 --top-k 20 --top-p 0.95 --min-p 0.0"

# ========================================================================================
# Main script
# ========================================================================================

docker run -d --name llamacpp --hostname openai_backend \
  --restart unless-stopped --ipc=host \
  --runtime nvidia --gpus all \
  -v $MODELS:/models \
  -p 8000:8000 \
  --network millegrille_net \
  $IMG --server \
  --offline -m "/models/$MODEL" \
  --port 8000 --host 0.0.0.0 \
  --no-slots $PARAMS ${LOGGING_PARAMS}
