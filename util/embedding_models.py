import os
from functools import lru_cache
from typing import Iterable, List

import numpy as np
from sentence_transformers import SentenceTransformer

DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def _get_model() -> SentenceTransformer:
    model_name = os.getenv("BERT_MODEL_NAME") or os.getenv("SENTENCE_MODEL")
    model_name = (model_name or DEFAULT_MODEL_NAME).strip()
    return SentenceTransformer(model_name)


def embed_sentences(sentences: Iterable[str]) -> np.ndarray:
    model = _get_model()
    embeddings = model.encode(
        list(sentences),
        batch_size=32,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return embeddings


def embed_query(query: str) -> np.ndarray:
    model = _get_model()
    vector = model.encode(
        [query],
        batch_size=1,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return vector[0]


def cosine_similarity(matrix: np.ndarray, vector: np.ndarray) -> np.ndarray:
    if matrix.size == 0:
        return np.zeros(0)
    return np.clip(matrix @ vector, -1.0, 1.0)


def batched_query_similarity(matrix: np.ndarray, queries: List[str]) -> np.ndarray:
    if matrix.size == 0:
        return np.zeros((len(queries), 0))
    model = _get_model()
    query_embeddings = model.encode(
        queries,
        batch_size=len(queries),
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return np.clip(query_embeddings @ matrix.T, -1.0, 1.0)
