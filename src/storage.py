import json
import pickle
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional
import numpy as np
from rank_bm25 import BM25Plus
from sentence_transformers import SentenceTransformer
from src.parser import DocumentChunk

@dataclass
class CardIndexData:
    card_id: str
    chunks: List[DocumentChunk]
    bm25: BM25Plus
    embeddings: np.ndarray

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
