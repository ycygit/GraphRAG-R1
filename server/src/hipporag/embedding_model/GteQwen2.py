from typing import List, Optional

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from ..utils.config_utils import BaseConfig
from ..utils.logging_utils import get_logger
from .base import BaseEmbeddingModel, EmbeddingConfig

logger = get_logger(__name__)


class Qwen2EmbeddingModel(BaseEmbeddingModel):
    """
    Embedding model subclass using sentence-transformers for Alibaba-NLP/gte-Qwen2-1.5B-instruct.
    """

    def __init__(
        self,
        global_config: Optional[BaseConfig] = None,
        embedding_model_name: Optional[str] = None
    ) -> None:
        super().__init__(global_config=global_config)
        self._init_embedding_config()
        # Set model name (default to Qwen2 instruct)
        self.embedding_model_name = embedding_model_name or "Alibaba-NLP/gte-Qwen2-1.5B-instruct"
        logger.debug(f"Using sentence-transformers model: {self.embedding_model_name}")

        # Initialize sentence-transformers model
        self.embedding_model = SentenceTransformer(self.embedding_model_name)
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()

        # # Configure normalization flag
        # self.norm = self.global_config.embedding_return_as_normalized
    def _init_embedding_config(self) -> None:
        """
        Extract embedding model-specific parameters to init the EmbeddingConfig.
        
        Returns:
            None
        """

        config_dict = {
            "embedding_model_name": self.embedding_model_name,
            "norm": self.global_config.embedding_return_as_normalized,
            # "max_seq_length": self.global_config.embedding_max_seq_len,
            "model_init_params": {
                # "model_name_or_path": self.embedding_model_name2mode_name_or_path[self.embedding_model_name],
                "pretrained_model_name_or_path": self.embedding_model_name,
                "trust_remote_code": True,
                'device_map': "auto",  # added this line to use multiple GPUs
                "torch_dtype": self.global_config.embedding_model_dtype,
                # **kwargs
            },
            "encode_params": {
                "max_length": self.global_config.embedding_max_seq_len,  # 32768 from official example,
                "instruction": "",
                "batch_size": self.global_config.embedding_batch_size,
                "num_workers": 32
            },
        }

        self.embedding_config = EmbeddingConfig.from_dict(config_dict=config_dict)
        logger.debug(f"Init {self.__class__.__name__}'s embedding_config: {self.embedding_config}")
    def batch_encode(self, texts: List[str], **kwargs) -> np.ndarray:
        """
        Encode a batch of texts into embeddings using sentence-transformers.
        Supports optional instruction prefix.
        """
        if isinstance(texts, str):
            texts = [texts]

        # Handle instruction formatting
        if "instruction" in kwargs and kwargs["instruction"]:
            instruction = kwargs["instruction"]
            texts = [f"Instruct: {instruction}\nQuery: {t}" for t in texts]

        # # Determine batch size
        # batch_size = getattr(self.global_config, "embedding_batch_size")
        batch_size = 8
        logger.debug(f"Qwen2SentenceEmbeddingModel encoding {len(texts)} texts with batch size {batch_size}")
        # Use sentence-transformers encode
        embeddings = self.embedding_model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
        )

        # Normalize if required
        if self.embedding_config.norm:
            embeddings = (embeddings.T / np.linalg.norm(embeddings, axis=1)).T

        return embeddings
