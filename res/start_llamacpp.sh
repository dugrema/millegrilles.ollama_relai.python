#!/bin/env bash

MODELS="/mnt/tmphold/tas/models"

IMG="registry.millegrilles.com/ggml-org/llama.cpp:server-cuda-5966"

PARAMS="--n-gpu-layers 99 --ctx-size 4096 --n-predict 2048"

MODEL="gguf/gemma-3-4b-it-qat-q4_0.gguf"
PARAMS="--n-gpu-layers 99 --mmproj /models/gguf/gemma-3-4B-mmproj-BF16.gguf --ctx-size 10240 --n-predict 2048 --ubatch-size 256 --batch-size 1024"

docker run --rm -d --name llamacpp --hostname openai_backend \
  --runtime nvidia --gpus all \
  -v $MODELS:/models \
  -p 8000:8000 \
  --network millegrille_net \
  $IMG \
  --offline -m "/models/$MODEL" \
  --port 8000 --host 0.0.0.0 \
  $PARAMS
