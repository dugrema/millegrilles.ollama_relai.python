#!/bin/env bash

IMG="vllm/vllm-openai:v0.9.2"
#IMG="public.ecr.aws/q9t5s3a7/vllm-cpu-release-repo:v0.9.2"

#MODEL="/root/vllm/models/gemma-3-4b-it"
MODEL="/root/vllm/models/gemma-3-4b-it-qat-q4_0-unquantized"
#MODEL="/root/vllm/models/gemma3-12b-it-qat_unquantized"
#MODEL="/root/vllm/models/deepseek-r1-0528-qwen3-8B"
#MODEL="/root/vllm/models/llama3.2-instruct"
#MODEL="/root/vllm/models/gemma-3n-E4B-it"
#MODEL="/root/vllm/models/gemma-3n-E2B-it"

docker run --runtime nvidia --gpus all --rm --name vllm --hostname vllm \
    -p 8001:8000 \
    -v ~/llm/huggingface:/root/.cache/huggingface \
    -v /mnt/tas/users/mathieu/lib/vllm:/root/vllm \
    --network millegrille_net \
    --ipc=host \
    --env "TORCH_CUDA_ARCH_LIST=8.0 8.6 8.7" \
    $IMG --model "${MODEL}" --max-num-seqs 1 --disable-log-requests \
    --chat-template-content-format openai --generation-config vllm \
    --quantization bitsandbytes --max-model-len 32K --max-seq-len-to-capture 2048 --gpu-memory-utilization 0.77 \
    --limit-mm-per-prompt '{"images": 1, "videos": 0}'

# gemma3n-E4B
#    --cpu-offload-gb 1.0 --quantization bitsandbytes --max-model-len 8K --gpu-memory-utilization 0.95 --enforce-eager
#    --cpu-offload-gb 1.0 --quantization bitsandbytes --max-model-len 10K --gpu-memory-utilization 0.95 --enforce-eager

# gemma3n-E2B:
#    --quantization bitsandbytes --max-model-len 16K --max-seq-len-to-capture 4096 --gpu-memory-utilization 0.83
#    --quantization bitsandbytes --max-model-len 4K --gpu-memory-utilization 0.9 --enforce-eager
#    --quantization bitsandbytes --max-model-len 30K --gpu-memory-utilization 0.95 --enforce-eager

# gemma3 4b qat
#    --quantization bitsandbytes --max-model-len 32K --max-seq-len-to-capture 2048 --gpu-memory-utilization 0.77 \
#    --limit-mm-per-prompt '{"images": 1, "videos": 0}'

# gemma3 4b - not working
#    --chat-template-content-format openai --generation-config vllm \
#    --quantization bitsandbytes --max-model-len 4K --gpu-memory-utilization 0.75

#    --dtype bfloat16 --cpu-offload-gb 10
#    --device cpu --max-model-len 4096 --max-num-seqs 1 --quantization bitsandbytes
#    --max-model-len 4096 --max-num-seqs 2 --gpu-memory-utilization 0.85 --quantization bitsandbytes
#    --cpu-offload-gb 4
#    --max-model-len 4096 --max-num-seqs 1 --gpu-memory-utilization 0.85
#    --max-model-len 4096 --max-num-seqs 1 --gpu-memory-utilization 0.8 --enforce-eager
#    --model "${MODEL}" --quantization bitsandbytes
#    --max-model-len 16384 --gpu-memory-utilization 0.69

# gemma-3-4b: --max-model-len 32768 --gpu-memory-utilization 0.85
# gemma-3-4b: --max-model-len 16384 --gpu-memory-utilization 0.69
# gemma-3-4b-qat_unquantized: --bitsandbytes --max-model-len 10240 --gpu-memory-utilization 0.75
# Deepseek R1 0528: --quantization bitsandbytes --max-model-len 8K --gpu-memory-utilization 0.98 --enforce-eager \
# ollama 3.2: --quantization bitsandbytes --max-model-len 12K --gpu-memory-utilization 0.75 \
