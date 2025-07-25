#!/bin/env bash

docker run --rm --name llamacpp \
  --runtime nvidia --gpus all \
  -v $MODELS:/models \
  -p 8000:8000 --network millegrille_net \
  $IMG \
  --offline -m "/models/$MODEL" \
  --port 8000 --host 0.0.0.0 \
  $PARAMS
