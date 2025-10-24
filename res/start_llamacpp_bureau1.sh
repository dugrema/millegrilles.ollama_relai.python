#!/bin/env bash

MODELS="/home/mathieu/tas/models"
MODELS_NVME="/home/mathieu/models"
MODELS_SSD240="/mnt/ssd240/models"

IMG="registry.millegrilles.com/ggml-org/llama.cpp:full-cuda_6812"

# LOGGING_PARAMS="--log-verbosity -1"
LOGGING_PARAMS="--log-timestamps --log-colors on --log-verbosity 0"

# ===============
# GPT-OSS
# ===============
GPTOSS_PARAMS="--temp 1.0 --top-p 1.0 --jinja"
# MODEL="/models_ssd240/gguf/gpt-oss-20b-FULL.gguf"
# PARAMS="-a chat --ctx-size 65536 ${GPTOSS_PARAMS}"

# ===============
# Qwen3
# ===============
QWEN3_PARAMS="-fa 1 --temp 0.6 --top-k 20 --top-p 0.95 --min-p 0 --repeat-penalty 1.05 --jinja"
MODEL="/models_ssd240/gguf/Qwen3-30B-A3B-Thinking-2507-UD-Q4_K_XL.gguf"
PARAMS="--n-cpu-moe 8 --ctx-size 8192 --ubatch-size 64 --batch-size 1024 ${QWEN3_PARAMS}"
# PARAMS="--n-cpu-moe 11 --ctx-size 16384 --ubatch-size 128 ${QWEN3_PARAMS}"

# ===============
# Gemma 3
# ===============
GEMMA3_PARAMS="--temp 1.0 --top-k 64 --top-p 0.95 --min-p 0.0 --jinja"
# MODEL="/models_ssd240/gguf/gemma-3-12b-it-UD-Q6_K_XL.gguf"
# PARAMS="-a vision --mmproj /models_ssd240/gguf/gemma-3-12B-mmproj-BF16.gguf --ubatch-size 256 --ctx-size 32768 ${GEMMA3_PARAMS}"
# MODEL="/models/gguf/gemma-3-27b-it-UD-Q2_K_XL.gguf"
# PARAMS="--ubatch-size 256 --ctx-size 16384 --mmproj /models/gguf/gemma-3-27B-mmproj-BF16.gguf ${GEMMA3_PARAMS}"
# MODEL="/models/gguf/gemma-3-27b-it-UD-Q3_K_XL.gguf"
# PARAMS="--ubatch-size 128 --ctx-size 16384 ${GEMMA3_PARAMS}"

# ===============
# Magistral
# ===============
MAGISTRAL_PARAMS="-fa 1 --temp 0.7 --top-p 0.95 --jinja"
# MODEL="/models/gguf/Magistral-Small-2509-UD-Q3_K_XL.gguf"
# PARAMS="--ubatch-size 256 --ctx-size 16384 ${MAGISTRAL_PARAMS}"
# PARAMS="--ctx-size 16384 ${MAGISTRAL_PARAMS}"

# ===============
# Coding models
# ===============
# MODEL="/models/gguf/GLM-4-32B-0414-UD-Q2_K_XL.gguf"
# PARAMS="-fa 0 --ubatch-size 128 --ctx-size 32768"
# MODEL="/models/gguf/ERNIE-4.5-21B-A3B-Thinking-UD-Q6_K_XL.gguf"
# PARAMS="-fa 1 --n-cpu-moe 10 --ubatch-size 128 --ctx-size 32768 --jinja"
# MODEL="/models_ssd240/gguf/Qwen3-Coder-30B-A3B-Instruct-UD-Q4_K_XL.gguf"
# PARAMS="--n-cpu-moe 8 --ctx-size 8192 --ubatch-size 64 --batch-size 1024 ${QWEN3_PARAMS}"
# PARAMS="--n-cpu-moe 15 --ctx-size 32768 --ubatch-size 256 ${QWEN3_PARAMS}"
# PARAMS="--n-cpu-moe 25 --ctx-size 65536 --ubatch-size 256 ${QWEN3_PARAMS}"
# PARAMS="--n-cpu-moe 45 --ctx-size 131072 --ubatch-size 256 ${QWEN3_PARAMS}"

# ================================================================================
# Main script
# ================================================================================

# Stop existing container
docker stop llamacpp 2> /dev/null
docker wait llamacpp 2> /dev/null
sleep 1

# Start
docker run -d --rm --name llamacpp --hostname openai_backend \
  --runtime nvidia --gpus all --ipc=host \
  -v $MODELS:/models -v $MODELS_NVME:/models_nvme -v $MODELS_SSD240:/models_ssd240 \
  -p 8000:8000 --network millegrille_net \
  $IMG \
  --server --offline --model "$MODEL" \
  --port 8000 --host 0.0.0.0 \
  $PARAMS $LOGGING_PARAMS

sleep 1
docker logs -f llamacpp

#  --no-webui
#  --port 8000 --host 0.0.0.0 \