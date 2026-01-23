### HippoRAG Retrieval Server

This directory contains a simple FastAPI-based retrieval server for HippoRAG.

### How to start the server

1. **Install dependencies**

```bash
pip install -r requirements.txt
```

2. **Load environment variables**

```bash
source setup_env.sh
```

3. **Start the server**

```bash
python server.py
```

By default the server runs at `http://127.0.0.1:8090`.  
You can change host/port and other settings via `setup_env.sh`.

---

### Configuration in `setup_env.sh`

The script `setup_env.sh` defines all required environment variables.  
You can edit it before running `source setup_env.sh`.

- **`RERANK_API_KEY`**  
  - API key for the rerank service (e.g. DashScope / Qwen rerank).  
  - Must be a valid key if you use reranking.  
  - Example: `export RERANK_API_KEY="sk-xxxx..."`.

- **`HF_ENDPOINT`**  
  - Hugging Face mirror endpoint.  
  - Default: `https://hf-mirror.com` (useful in mainland China).  
  - Change to `https://huggingface.co` if you don't want to use the mirror.

- **`CUDA_VISIBLE_DEVICES`**  
  - GPU device IDs to use, e.g. `"0"` or `"0,1"`.  
  - Default in this script: `2`.  
  - Set to an empty string if you want to force CPU: `export CUDA_VISIBLE_DEVICES=""`.

- **`LLM_MODEL_NAME`**  
  - Logical name of the LLM used by HippoRAG (for logging/config).  
  - Default: `qwen7B_4096:latest`.  
  - Should match what your LLM service actually provides.

- **`LLM_BASE_URL`**  
  - Base URL of the LLM inference server.  
  - Default: `http://localhost:11434/`.  
  - Change this if your LLM server runs on a different host/port.

- **`EMBEDDING_MODEL_NAME`**  
  - Name of the embedding model used to index and retrieve documents.  
  - Default: `nvidia/NV-Embed-v2`.  
  - Replace with another supported embedding model if needed.

- **`RERANK_BASE_URL`**  
  - Base URL of the rerank HTTP API.  
  - Default: `https://dashscope.aliyuncs.com/compatible-mode/v1`.

- **`RERANK_MODEL`**  
  - Rerank model name used at `RERANK_BASE_URL`.  
  - Default: `qwen-turbo-latest`.

- **`DATA_PATH`**  
  - Path to the input data file (JSON) relative to the project root.  
  - Default: `outputs/server/openie_results_ner_qwen7B_4096:latest.json`.  
  - The server will load documents from this file at startup.

- **`SAVE_DIR`**  
  - Directory where indices and intermediate files are stored.  
  - Default: `outputs/server`.

- **`SERVER_HOST`**  
  - Host/IP for the FastAPI server.  
  - Default: `127.0.0.1` (only accessible locally).  
  - Set to `0.0.0.0` to expose the service to other machines.

- **`SERVER_PORT`**  
  - TCP port for the server.  
  - Default: `8090`.

- **`LOG_LEVEL`**  
  - Logging level for the server.  
  - Default: `info`.  
  - Other common values: `debug`, `warning`, `error`.

After editing `setup_env.sh`, run:

```bash
source setup_env.sh
python server.py
```

to start the service with your customized configuration.


