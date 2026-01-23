"""
HippoRAG retrieval service API
Provides HippoRAG-based indexing and query service
"""
import os
import json
import logging
from typing import List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.hipporag.utils.config_utils import BaseConfig
from src.hipporag import HippoRAG
from config import (
    HF_ENDPOINT, CUDA_VISIBLE_DEVICES,
    LLM_MODEL_NAME, EMBEDDING_MODEL_NAME, LLM_BASE_URL,
    SAVE_DIR, DATA_PATH, HIPPORAG_CONFIG,
    SERVER_HOST, SERVER_PORT, LOG_LEVEL
)

# Set environment variables
os.environ["HF_ENDPOINT"] = HF_ENDPOINT
os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES

# Configure logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL.upper()),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="HippoRAG2 Retrieval Service",
    description="",
    version="1.0.0"
)

# Initialize HippoRAG config
config = BaseConfig(**HIPPORAG_CONFIG)

# Initialize HippoRAG instance
hipporag = HippoRAG(
    global_config=config,
    save_dir=SAVE_DIR,
    llm_model_name=LLM_MODEL_NAME,
    embedding_model_name=EMBEDDING_MODEL_NAME,
    llm_base_url=LLM_BASE_URL,
)

logger.info("HippoRAG2 instance initialized successfully")

class QARequest(BaseModel):
    """Query request model"""
    query: List[str] = Field(..., description="Query list", min_items=1)
    retrieval_num: int = Field(default=5, description="Retrieval return document number", ge=1, le=20)


class QueryResponse(BaseModel):
    """Single query response model"""
    docs: List[str] = Field(default_factory=list, description="Retrieved documents list")
    facts: List[tuple] = Field(default_factory=list, description="Extracted facts list as tuples")


class QAResponse(BaseModel):
    """Batch query response model"""
    result: List[QueryResponse]


@app.on_event("startup")
async def load_data_and_index():
    """Load data and build index on startup"""
    try:
        logger.info(f"Loading data: {DATA_PATH}")
        
        if not os.path.exists(DATA_PATH):
            raise FileNotFoundError(f"Data file not found: {DATA_PATH}")
        
        with open(DATA_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        docs = [e["passage"] for e in data.get("docs", []) if "passage" in e]
        
        if not docs:
            raise ValueError("No valid documents found")
        
        logger.info(f"Successfully loaded {len(docs)} documents, starting to build index...")
        hipporag.index(docs=docs)
        logger.info("Index building completed")
        
    except Exception as e:
        logger.error(f"Data loading or index building failed: {e}")
        raise


@app.post("/query", response_model=QAResponse)
async def rag_qa(request: QARequest):
    """
    Retrieval query endpoint
    
    Args:
        request: request with query list
        
    Returns:
        Batch results with retrieved docs and facts
    """
    try:
        logger.info(f"Received {len(request.query)} query requests")
        
        queries = hipporag.rag_qa(
            queries=request.query,
            only_retrive=True,
            retrieval_num=request.retrieval_num
        )
        
        results = []
        for query_result in queries:
            docs = query_result.docs or []
            facts = query_result.facts or []

            results.append(QueryResponse(
                docs=docs,
                facts=facts
            ))
        
        logger.info(f"Successfully processed {len(results)} queries")
        return QAResponse(result=results)
        
    except Exception as e:
        logger.error(f"Query processing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "ok", "service": "HippoRAG2 Retrieval Service"}


@app.get("/stats")
async def get_stats():
    """Get service stats"""
    try:
        return {
            "save_dir": SAVE_DIR,
            "llm_model": LLM_MODEL_NAME,
            "embedding_model": EMBEDDING_MODEL_NAME,
            "config": HIPPORAG_CONFIG
        }
    except Exception as e:
        logger.error(f"Failed to get statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    
    logger.info(f"Starting HippoRAG2 Retrieval Service...")
    logger.info(f"Service address: http://{SERVER_HOST}:{SERVER_PORT}")
    logger.info(f"Data path: {DATA_PATH}")
    
    uvicorn.run(
        app,
        host=SERVER_HOST,
        port=SERVER_PORT,
        log_level=LOG_LEVEL
    )