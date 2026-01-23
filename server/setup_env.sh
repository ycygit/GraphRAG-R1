#!/bin/bash

# HippoRAG retrieval service environment setup script
# Usage: source setup_env.sh

echo "======================================"
echo "  HippoRAG Environment variables setup"
echo "======================================"

# ========== Required ==========
# ⚠️ Please set your API key below
export RERANK_API_KEY="${RERANK_API_KEY:-sk-your-api-key-here}"

# ========== Optional ==========

# HuggingFace mirror endpoint
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

# GPU device IDs (comma-separated, e.g. "0,1")
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"

# LLM model settings
export LLM_MODEL_NAME="${LLM_MODEL_NAME:-qwen7B_4096:latest}"
export LLM_BASE_URL="${LLM_BASE_URL:-http://localhost:11434/}"

# Embedding model settings
export EMBEDDING_MODEL_NAME="${EMBEDDING_MODEL_NAME:-nvidia/NV-Embed-v2}"

# Rerank model settings
export RERANK_BASE_URL="${RERANK_BASE_URL:-https://dashscope.aliyuncs.com/compatible-mode/v1}"
export RERANK_MODEL="${RERANK_MODEL:-qwen-turbo-latest}"

# Data path (relative to project root)
export DATA_PATH="${DATA_PATH:-outputs/server/openie_results_ner_qwen7B_4096:latest.json}"

# Index save directory
export SAVE_DIR="${SAVE_DIR:-outputs/server}"

# Server settings
export SERVER_HOST="${SERVER_HOST:-127.0.0.1}"
export SERVER_PORT="${SERVER_PORT:-8090}"
export LOG_LEVEL="${LOG_LEVEL:-info}"

# ========== Confirmation ==========
echo ""
echo "已设置以下环境变量："
echo "  HF_ENDPOINT = $HF_ENDPOINT"
echo "  CUDA_VISIBLE_DEVICES = $CUDA_VISIBLE_DEVICES"
echo "  LLM_MODEL_NAME = $LLM_MODEL_NAME"
echo "  LLM_BASE_URL = $LLM_BASE_URL"
echo "  EMBEDDING_MODEL_NAME = $EMBEDDING_MODEL_NAME"
echo "  RERANK_API_KEY = ${RERANK_API_KEY:0:10}..." # show only first 10 chars
echo "  RERANK_BASE_URL = $RERANK_BASE_URL"
echo "  RERANK_MODEL = $RERANK_MODEL"
echo "  DATA_PATH = $DATA_PATH"
echo "  SAVE_DIR = $SAVE_DIR"
echo "  SERVER_HOST = $SERVER_HOST"
echo "  SERVER_PORT = $SERVER_PORT"
echo "  LOG_LEVEL = $LOG_LEVEL"
echo ""

# Validate required fields
if [ "$RERANK_API_KEY" = "sk-your-api-key-here" ] || [ -z "$RERANK_API_KEY" ]; then
    echo "⚠️  Warning: RERANK_API_KEY is not set or using default value"
    echo "   Please set RERANK_API_KEY in the script or via shell:"
    echo "   export RERANK_API_KEY='your-actual-api-key'"
    echo ""
fi

echo "======================================"
echo "Environment variables setup completed!"
echo ""
echo "Usage:"
echo "  source setup_env.sh    # load env vars"
echo "  python server.py       # start server"
echo "======================================"
