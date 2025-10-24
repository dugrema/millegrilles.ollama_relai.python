#!/bin/env bash

MODELS="/home/mathieu/models"

# IMG="registry.millegrilles.com/ggml-org/llama.cpp:full-cuda_6692"
# IMG="registry.millegrilles.com/ggml-org/llama.cpp:full-cuda_6767"
IMG="registry.millegrilles.com/ggml-org/llama.cpp:full-cuda_6812"

# LOGGING_PARAMS="--log-verbosity -1"
LOGGING_PARAMS="--log-timestamps --log-colors on --log-verbosity 0"

PARAMS="--n-gpu-layers 99 --ctx-size 8192 --ubatch-size 256 --batch-size 2048 --n-predict 4096 --jinja"

# Deepseek R1 0528 Qwen3
# MODEL="/models/gguf/DeepSeek-R1-0528-Qwen3-8B-UD-Q4_K_XL.gguf"
# PARAMS="--ubatch-size 128 --batch-size 4096 --ctx-size 16384 --n-predict 8192 --temp 0.6 --top-p 0.95 --jinja"

# Gemma 3
GEMMA3_PARAMS="--temp 1.0 --top-k 64 --top-p 0.95 --min-p 0.0 --jinja"

MODEL="/models/gguf/gemma-3-4b-it-UD-Q6_K_XL.gguf"
GEMMA3_4B_MMPROJ="--mmproj /models/gguf/gemma-3-4b-mmproj-BF16.gguf"
PARAMS="-a vision -fa 1 -t 4 --ubatch-size 512 --batch-size 8192 --ctx-size 65536 ${GEMMA3_PARAMS} ${GEMMA3_4B_MMPROJ}"

# MODEL="/models/gguf/gemma-3-12b-it-UD-Q2_K_XL.gguf"
# PARAMS="--swa-checkpoints 1 -fa 1 -t 4 --mmproj /models/gguf/gemma-3-12B-mmproj-BF16.gguf --ubatch-size 256 --batch-size 256 --ctx-size 8192 --temp 1.0 --top-k 64 --top-p 0.95 --min-p 0.0 --jinja"
# PARAMS="--no-kv-offload --swa-checkpoints 1 -fa 1 -t 4 --mmproj /models/gguf/gemma-3-12B-mmproj-BF16.gguf --ubatch-size 256 --batch-size 256 --ctx-size 8192 --temp 1.0 --top-k 64 --top-p 0.95 --min-p 0.0 --jinja"
# PARAMS="--n-gpu-layers 35 --swa-checkpoints 1 -fa 1 -t 4 --mmproj /models/gguf/gemma-3-12B-mmproj-BF16.gguf --ubatch-size 256 --batch-size 256 --ctx-size 32768 --temp 1.0 --top-k 64 --top-p 0.95 --min-p 0.0 --jinja"

# Stop existing container
docker stop llamacpp 2> /dev/null; docker rm llamacpp 2> /dev/null

# Start
docker run -d --name llamacpp --hostname openai_backend \
  --restart unless-stopped \
  --runtime nvidia --gpus all --ipc=host \
  -v $MODELS:/models \
  -p 8000:8000 \
  --network millegrille_net \
  $IMG \
  --server --offline --model "$MODEL" \
  --port 8000 --host 0.0.0.0 \
  $PARAMS $LOGGING_PARAMS
