#!/bin/env bash

MODELS_PATH="/var/opt/vllm"

IMG="registry.millegrilles.com/vllm/vllm-openai:v0.9.2"

#MODEL="google/gemma-3-4b-it-qat-q4_0-unquantized"
MODEL="/m/gemma-3-4b-it-qat-q4_0-unquantized"
TORCH_CUDA_ARCH_LIST="8.0 8.6 8.7"

# Optional flags
# -p 8001:8000 \

docker run -d --runtime nvidia --gpus all --name vllm --hostname vllm \
    --restart unless-stopped --ipc=host \
    -v "${MODELS_PATH}/cache/huggingface:/root/.cache/huggingface" \
    -v "${MODELS_PATH}/models:/m" \
    --env "TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}" \
    --network millegrille_net \
    $IMG \
    --model "${MODEL}" --max-num-seqs 1 --disable-log-requests \
    --quantization bitsandbytes --max-model-len 10K --max-seq-len-to-capture 4096 --gpu-memory-utilization 0.91

# gemma3-4b-qat Using nvidia graph for speedup
# --quantization bitsandbytes --max-model-len 10K --max-seq-len-to-capture 4096 --gpu-memory-utilization 0.91
# gemma3-4b-qat Maximum context, no graph
# --quantization bitsandbytes --max-model-len 30K --gpu-memory-utilization 0.98 --enforce-eager
