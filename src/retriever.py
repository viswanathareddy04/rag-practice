import re
from dataclasses import dataclass
from typing import List, Optional
import numpy as np
from src.parser import DocumentChunk
from src.storage import StorageManager

@dataclass
class ScoredChunk:
    chunk: DocumentChunk
    score: float
    retrieval_method: str

class HybridRetriever:
    """Performs card-scoped hybrid retrieval combining BM25 lexical search and dense cosine similarity via RRF."""

    def __init__(self, storage_manager: StorageManager, rrf_k: int = 60):
        self.storage_manager = storage_manager
        self.rrf_k = rrf_k

    def retrieve(self, card_id: str, query: str, top_k: int = 5) -> List[ScoredChunk]:
        index_data = self.storage_manager.load_card_index(card_id)
        chunks = index_data.chunks

        if not chunks:
            return []

        # 1. BM25 Lexical Retrieval with regex tokenization
        tokenized_query = re.findall(r"\w+", query.lower())
        bm25_scores = np.array(index_data.bm25.get_scores(tokenized_query))
        bm25_ranked_indices = np.argsort(bm25_scores)[::-1]

        # 2. Dense Vector Retrieval
        query_embedding = self.storage_manager.embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        dense_scores = np.dot(index_data.embeddings, query_embedding.T).flatten()
        dense_ranked_indices = np.argsort(dense_scores)[::-1]

        # 3. Reciprocal Rank Fusion (RRF)
        rrf_scores: dict[int, float] = {}

        # Evaluate top candidates
        eval_depth = min(20, len(chunks))
        for rank, idx in enumerate(bm25_ranked_indices[:eval_depth]):
            if bm25_scores[idx] > 0:
                rrf_scores[idx] = rrf_scores.get(idx, 0.0) + (1.0 / (self.rrf_k + rank + 1))

        for rank, idx in enumerate(dense_ranked_indices[:eval_depth]):
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + (1.0 / (self.rrf_k + rank + 1))

        sorted_indices = sorted(rrf_scores.keys(), key=lambda i: rrf_scores[i], reverse=True)

        results: List[ScoredChunk] = []
        for idx in sorted_indices[:top_k]:
            results.append(ScoredChunk(
                chunk=chunks[idx],
                score=float(rrf_scores[idx]),
                retrieval_method="hybrid_rrf"
            ))

        return results
