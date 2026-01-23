"""
HippoRAG retrieval service configuration
"""
import os

# Environment variables
HF_ENDPOINT = os.getenv("HF_ENDPOINT", "https://hf-mirror.com")
CUDA_VISIBLE_DEVICES = os.getenv("CUDA_VISIBLE_DEVICES", "0")

# Model settings
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "qwen7B_4096:latest")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "nvidia/NV-Embed-v2")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:11434/")

# Data path settings
SAVE_DIR = os.getenv("SAVE_DIR", "outputs/server")
DATA_PATH = os.getenv(
    "DATA_PATH",
    "outputs/server/openie_results_ner_qwen7B_4096:latest.json"
)

# HippoRAG settings
HIPPORAG_CONFIG = {
    "rerank_dspy_file_path": "src/hipporag/prompts/dspy_prompts/filter_llama3.3-70B-Instruct.json",
    "retrieval_top_k": 5,
    "linking_top_k": 20,
    "max_qa_steps": 3,
    "qa_top_k": 5,
    "graph_type": "facts_and_sim_passage_node_unidirectional",
    "embedding_batch_size": 8,
    "max_new_tokens": None,
}

# Rerank model settings (for KG fact filtering)
RERANK_API_KEY = os.getenv("RERANK_API_KEY", "")
RERANK_BASE_URL = os.getenv("RERANK_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
RERANK_MODEL = os.getenv("RERANK_MODEL", "qwen-turbo-latest")

SERVER_HOST = os.getenv("SERVER_HOST", "0.0.0.0")
SERVER_PORT = int(os.getenv("SERVER_PORT", "8090"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "info")

os.environ["HF_ENDPOINT"] = HF_ENDPOINT
os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES
os.environ["RERANK_API_KEY"] = RERANK_API_KEY
os.environ["RERANK_BASE_URL"] = RERANK_BASE_URL
os.environ["RERANK_MODEL"] = RERANK_MODEL
