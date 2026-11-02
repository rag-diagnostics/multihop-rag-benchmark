#!/usr/bin/env bash
# Example only: execute each command in a SEPARATE terminal / GPU environment.
# Install vLLM separately with a version compatible with your GPUs and Python.
# Ensure upstream licence restrictions have been clarified before dataset generation.
# Avoid binding model servers to 0.0.0.0 on shared networks without access controls.

# Terminal 1, GPU 0:
# export CUDA_VISIBLE_DEVICES=0
# vllm serve meta-llama/Llama-3.1-8B-Instruct --host 127.0.0.1 --port 8000 --max-model-len 8192

# Terminal 2, GPU 1:
# export CUDA_VISIBLE_DEVICES=1
# vllm serve Qwen/Qwen2.5-7B-Instruct --host 127.0.0.1 --port 8001 --max-model-len 4096

# Only after both are ready:
# python scripts/check_local_endpoints.py
# export OPENAI_API_KEY=local-vllm-placeholder
# python scripts/generate_model_configs.py --games 1 --output /tmp/one_game.txt
# python scripts/run_games.py --engine-dir /path/to/AI_Diplomacy \
#     --config /tmp/one_game.txt --output-dir /tmp/diplomacy_games --dry-run
