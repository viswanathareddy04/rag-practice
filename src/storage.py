import json
import os
import pickle
import re
import uuid
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Dict, Any
import numpy as np
import torch
from qdrant_client import QdrantClient, models
from rank_bm25 import BM25Plus
from sentence_transformers import SentenceTransformer

from src.parser import DocumentChunk, ParentSection, ChildChunk
from src.bm25_vectorizer import BM25Vectorizer

@dataclass
class CardIndexData:
    card_id: str
    chunks: List[DocumentChunk]
    bm25: BM25Plus
    embeddings: np.ndarray

class QdrantStorageManager:
    """Manages cloud/local Qdrant vector storage with named dense & BM25 sparse vectors."""

    COLLECTION_NAME = "insurance_chunks"

    def __init__(self, client: Optional[QdrantClient] = None, url: Optional[str] = None,
                 api_key: Optional[str] = None, model_name: str = "BAAI/bge-base-en-v1.5",
                 device: str = "auto"):
        self.model_name = os.getenv("EMBEDDING_MODEL", model_name)
        
        # Resolve compute device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self._embedder: Optional[SentenceTransformer] = None
        self.bm25_vectorizer = BM25Vectorizer()

        # Resolve Qdrant client
        if client is not None:
            self.client = client
        else:
            q_url = url or os.getenv("QDRANT_URL")
            q_api_key = api_key or os.getenv("QDRANT_API_KEY")

            if q_url and q_api_key:
                self.client = QdrantClient(url=q_url, api_key=q_api_key)
            elif os.getenv("QDRANT_LOCAL", "").lower() in ("true", "1"):
                self.client = QdrantClient(path="./.index/qdrant")
            else:
                self.client = QdrantClient(location=":memory:")

    @property
    def embedder(self) -> SentenceTransformer:
        if self._embedder is None:
            self._embedder = SentenceTransformer(self.model_name, device=self.device)
        return self._embedder

    def create_collection_if_not_exists(self):
        """Creates the insurance_chunks collection with named dense + sparse vectors and card_id payload index."""
        collections = self.client.get_collections().collections
        existing_names = [c.name for c in collections]

        if self.COLLECTION_NAME not in existing_names:
            dim = self.embedder.get_sentence_embedding_dimension()
            self.client.create_collection(
                collection_name=self.COLLECTION_NAME,
                vectors_config={
                    "dense": models.VectorParams(size=dim, distance=models.Distance.COSINE)
                },
                sparse_vectors_config={
                    "bm25": models.SparseVectorParams()
                }
            )
            # Create payload index on card_id for fast pre-filtering across 20-30 cards
            self.client.create_payload_index(
                collection_name=self.COLLECTION_NAME,
                field_name="card_id",
                field_schema=models.PayloadSchemaType.KEYWORD
            )

    def is_indexed(self, card_id: str) -> bool:
        """Returns True if at least one chunk exists for card_id in the collection."""
        collections = self.client.get_collections().collections
        if self.COLLECTION_NAME not in [c.name for c in collections]:
            return False

        res = self.client.count(
            collection_name=self.COLLECTION_NAME,
            count_filter=models.Filter(
                must=[models.FieldCondition(key="card_id", match=models.MatchValue(value=card_id))]
            )
        )
        return res.count > 0

    def save_card_chunks(self, card_id: str, parent_sections: List[ParentSection], child_chunks: List[ChildChunk]):
        """Idempotently saves ChildChunks and their ParentSections to Qdrant Cloud."""
        self.create_collection_if_not_exists()

        # 1. Idempotent deletion: remove any existing points for this card
        self.client.delete(
            collection_name=self.COLLECTION_NAME,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[models.FieldCondition(key="card_id", match=models.MatchValue(value=card_id))]
                )
            )
        )

        if not child_chunks:
            return

        # 2. Fit BM25 vectorizer on the card's child corpus
        child_texts = [c.content for c in child_chunks]
        self.bm25_vectorizer.fit(child_texts)

        # 3. Dense embeddings
        dense_embeddings = self.embedder.encode(child_texts, convert_to_numpy=True, normalize_embeddings=True)

        # 4. Assemble points
        points: List[models.PointStruct] = []
        for i, c in enumerate(child_chunks):
            # Deterministic UUID from child_id
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, c.child_id))
            dense_vec = dense_embeddings[i].tolist()
            sparse_vec = self.bm25_vectorizer.encode_document(c.content)

            payload = asdict(c)
            # Ensure bank metadata is present
            if "bank" not in payload:
                payload["bank"] = card_id.split("_")[0] if "_" in card_id else "unknown"

            points.append(models.PointStruct(
                id=point_id,
                vector={
                    "dense": dense_vec,
                    "bm25": sparse_vec
                },
                payload=payload
            ))

        # 5. Batch upsert into Qdrant
        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            self.client.upsert(collection_name=self.COLLECTION_NAME, points=batch)


class StorageManager:
    """Manages persistence of chunks, BM25 indices, and dense embeddings per credit card."""

    def __init__(self, storage_dir: Path, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name
        self._embedder: Optional[SentenceTransformer] = None

    @property
    def embedder(self) -> SentenceTransformer:
        if self._embedder is None:
            self._embedder = SentenceTransformer(self.model_name)
        return self._embedder

    def _get_card_dir(self, card_id: str) -> Path:
        return self.storage_dir / card_id

    def is_indexed(self, card_id: str) -> bool:
        card_dir = self._get_card_dir(card_id)
        return (card_dir / "chunks.json").exists() and \
               (card_dir / "bm25.pkl").exists() and \
               (card_dir / "embeddings.npy").exists()

    def tokenize(self, text: str) -> List[str]:
        return re.findall(r"\w+", text.lower())

    def save_card_index(self, card_id: str, chunks: List[DocumentChunk]):
        card_dir = self._get_card_dir(card_id)
        card_dir.mkdir(parents=True, exist_ok=True)

        chunks_data = [asdict(c) for c in chunks]
        with open(card_dir / "chunks.json", "w", encoding="utf-8") as f:
            json.dump(chunks_data, f, ensure_ascii=False, indent=2)

        # Build BM25Plus index with clean word tokens
        tokenized_corpus = [self.tokenize(c.full_text_with_context) for c in chunks]
        bm25 = BM25Plus(tokenized_corpus)
        with open(card_dir / "bm25.pkl", "wb") as f:
            pickle.dump(bm25, f)

        # Build dense embeddings
        texts_to_embed = [c.full_text_with_context for c in chunks]
        embeddings = self.embedder.encode(texts_to_embed, convert_to_numpy=True, normalize_embeddings=True)
        np.save(card_dir / "embeddings.npy", embeddings)

    def load_card_index(self, card_id: str) -> CardIndexData:
        card_dir = self._get_card_dir(card_id)
        if not self.is_indexed(card_id):
            raise FileNotFoundError(f"Index for card '{card_id}' does not exist.")

        with open(card_dir / "chunks.json", "r", encoding="utf-8") as f:
            chunks_data = json.load(f)
        chunks = [DocumentChunk(**item) for item in chunks_data]

        with open(card_dir / "bm25.pkl", "rb") as f:
            bm25 = pickle.load(f)

        embeddings = np.load(card_dir / "embeddings.npy")
        return CardIndexData(card_id=card_id, chunks=chunks, bm25=bm25, embeddings=embeddings)
