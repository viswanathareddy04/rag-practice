import re
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Set
import numpy as np
from qdrant_client import models

from src.parser import DocumentChunk, ParentSection
from src.storage import StorageManager, QdrantStorageManager

from concurrent.futures import ThreadPoolExecutor, as_completed

@dataclass
class ScoredChunk:
    chunk: DocumentChunk
    score: float
    retrieval_method: str

@dataclass
class RetrievedContext:
    parents: List[ParentSection]
    citations: List[Dict[str, Any]]

@dataclass
class MultiCardContext:
    card_contexts: Dict[str, RetrievedContext]  # card_id -> RetrievedContext
    card_names: Dict[str, str]                 # card_id -> display card name


CARD_ID_ALIASES: Dict[str, str] = {
    "td_aeroplan_vip": "td_aeroplan_visa_infinite_privilege_card_coi",
    "cibc_aeroplan_vip": "cibc_aeroplan_visa_infinite_privilege_card_insurance",
    "scotia_passport_vip": "passport_visa_inifite_privilege_card_certificate_of_insurance_2025_en",
    "td_aeroplan_visa_infinite_privilege_card_coi": "td_aeroplan_vip",
    "cibc_aeroplan_visa_infinite_privilege_card_insurance": "cibc_aeroplan_vip",
    "passport_visa_inifite_privilege_card_certificate_of_insurance_2025_en": "scotia_passport_vip",
}


class HybridParentRetriever:
    """Performs card-scoped hybrid retrieval combining BM25 sparse vectors and dense embeddings via Qdrant Cloud RRF,
    then expands top child matches into deduplicated ParentSections."""

    def __init__(self, storage_manager: QdrantStorageManager, rrf_k: int = 60):
        self.storage_manager = storage_manager
        self.client = storage_manager.client
        self.collection_name = storage_manager.COLLECTION_NAME
        self.rrf_k = rrf_k

    def retrieve(self, card_id: str, query: str, top_k_parents: int = 4, top_k_children: int = 15) -> RetrievedContext:
        """Executes hybrid dense + sparse search in Qdrant Cloud and resolves to ParentSections."""
        # 1. Encode dense query
        dense_query = self.storage_manager.embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0].tolist()

        # 2. Encode sparse query (BM25)
        sparse_query = self.storage_manager.bm25_vectorizer.encode_query(query)

        target_ids = [card_id]
        if card_id in CARD_ID_ALIASES:
            target_ids.append(CARD_ID_ALIASES[card_id])

        card_filter = models.Filter(
            must=[models.FieldCondition(key="card_id", match=models.MatchAny(any=target_ids))]
        )

        # 3. Single Qdrant Cloud hybrid query with RRF
        try:
            res = self.client.query_points(
                collection_name=self.collection_name,
                prefetch=[
                    models.Prefetch(query=dense_query, using="dense", filter=card_filter, limit=30),
                    models.Prefetch(query=sparse_query, using="bm25", filter=card_filter, limit=30),
                ],
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                limit=top_k_children
            )
            points = res.points
        except Exception:
            # Fallback if sparse vector query fails or is empty
            res = self.client.query_points(
                collection_name=self.collection_name,
                query=dense_query,
                using="dense",
                query_filter=card_filter,
                limit=top_k_children
            )
            points = res.points

        # 4. Deduplicate parents by parent_id while preserving rank
        parents: List[ParentSection] = []
        seen_parent_ids: Set[str] = set()
        citations: List[Dict[str, Any]] = []

        for p in points:
            payload = p.payload or {}
            parent_id = payload.get("parent_id")

            # Extract child citation
            citations.append({
                "child_id": payload.get("child_id"),
                "parent_id": parent_id,
                "breadcrumb": payload.get("breadcrumb", ""),
                "page_number": payload.get("page_number", 1),
                "content": payload.get("child_content", payload.get("content", "")),
                "score": getattr(p, "score", 0.0)
            })

            # Deduplicate parent
            if parent_id and parent_id not in seen_parent_ids and len(parents) < top_k_parents:
                seen_parent_ids.add(parent_id)
                parents.append(ParentSection(
                    parent_id=parent_id,
                    card_id=payload.get("card_id", card_id),
                    card_name=payload.get("card_name", ""),
                    policy_name=payload.get("policy_name", ""),
                    section_title=payload.get("section_title", ""),
                    page_start=payload.get("page_start", payload.get("page_number", 1)),
                    page_end=payload.get("page_end", payload.get("page_number", 1)),
                    content=payload.get("parent_content", payload.get("content", ""))
                ))

        return RetrievedContext(parents=parents, citations=citations)

    def retrieve_multicard(
        self,
        card_ids: List[str],
        query: str,
        top_k_parents_per_card: int = 4,
        top_k_children_per_card: int = 15
    ) -> MultiCardContext:
        """Executes parallel hybrid search across multiple cards in Qdrant Cloud."""
        card_contexts: Dict[str, RetrievedContext] = {}
        card_names: Dict[str, str] = {}

        def _fetch_card(cid: str):
            try:
                ctx = self.retrieve(
                    card_id=cid,
                    query=query,
                    top_k_parents=top_k_parents_per_card,
                    top_k_children=top_k_children_per_card
                )
            except Exception:
                ctx = RetrievedContext(parents=[], citations=[])
            c_name = ctx.parents[0].card_name if ctx.parents and ctx.parents[0].card_name else cid
            return cid, c_name, ctx


        with ThreadPoolExecutor(max_workers=min(len(card_ids), 3)) as executor:
            futures = [executor.submit(_fetch_card, cid) for cid in card_ids]
            for f in as_completed(futures):
                cid, c_name, ctx = f.result()
                card_contexts[cid] = ctx
                card_names[cid] = c_name

        return MultiCardContext(card_contexts=card_contexts, card_names=card_names)



class HybridRetriever:
    """Legacy: Performs card-scoped hybrid retrieval combining BM25 lexical search and dense cosine similarity via RRF."""

    STOP_WORDS = {
        "what", "is", "the", "for", "a", "an", "and", "or", "in", "on", "to",
        "of", "with", "at", "by", "do", "does", "how", "much", "many", "can", "i"
    }

    def __init__(self, storage_manager: StorageManager, rrf_k: int = 60):
        self.storage_manager = storage_manager
        self.rrf_k = rrf_k

    def retrieve(self, card_id: str, query: str, top_k: int = 8) -> List[ScoredChunk]:
        index_data = self.storage_manager.load_card_index(card_id)
        chunks = index_data.chunks

        if not chunks:
            return []

        # 1. BM25 Lexical Retrieval with regex tokenization and stopword filtering
        raw_tokens = re.findall(r"\w+", query.lower())
        tokenized_query = [t for t in raw_tokens if t not in self.STOP_WORDS]
        if not tokenized_query:
            tokenized_query = raw_tokens

        bm25_scores = np.array(index_data.bm25.get_scores(tokenized_query))
        bm25_ranked_indices = np.argsort(bm25_scores)[::-1]

        # 2. Dense Vector Retrieval
        query_embedding = self.storage_manager.embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        dense_scores = np.dot(index_data.embeddings, query_embedding.T).flatten()
        dense_ranked_indices = np.argsort(dense_scores)[::-1]

        # 3. Reciprocal Rank Fusion (RRF)
        rrf_scores: dict[int, float] = {}

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
