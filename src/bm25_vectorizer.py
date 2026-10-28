import math
import re
import zlib
from collections import Counter
from typing import List, Dict, Set
from qdrant_client import models

class BM25Vectorizer:
    STOP_WORDS: Set[str] = {
        "what", "is", "the", "for", "a", "an", "and", "or", "in", "on", "to",
        "of", "with", "at", "by", "do", "does", "how", "much", "many", "can", "i", "my"
    }

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_count = 0
        self.avg_doc_len = 0.0
        self.idf: Dict[str, float] = {}

    def tokenize(self, text: str) -> List[str]:
        return re.findall(r"\w+", text.lower())

    def hash_token(self, token: str) -> int:
        idx = zlib.crc32(token.encode("utf-8")) % (2**31 - 1)
        return idx if idx > 0 else 1

    def fit(self, corpus: List[str]):
        self.doc_count = len(corpus)
        doc_lengths = []
        df: Counter = Counter()

        for doc in corpus:
            tokens = self.tokenize(doc)
            doc_lengths.append(len(tokens))
            unique_tokens = set(tokens)
            df.update(unique_tokens)

        self.avg_doc_len = sum(doc_lengths) / max(1, self.doc_count)
        self.idf = {}
        for token, freq in df.items():
            self.idf[token] = math.log(1.0 + (self.doc_count - freq + 0.5) / (freq + 0.5))

    def encode_document(self, text: str) -> models.SparseVector:
        tokens = self.tokenize(text)
        doc_len = len(tokens)
        counts = Counter(tokens)

        indices = []
        values = []
        for token, count in counts.items():
            idf_val = self.idf.get(token, 1.0)
            numerator = count * (self.k1 + 1.0)
            denominator = count + self.k1 * (1.0 - self.b + self.b * (doc_len / max(1.0, self.avg_doc_len)))
            weight = idf_val * (numerator / max(1e-6, denominator))
            if weight > 0:
                indices.append(self.hash_token(token))
                values.append(float(weight))

        return models.SparseVector(indices=indices, values=values)

    def encode_query(self, query: str) -> models.SparseVector:
        raw_tokens = self.tokenize(query)
        filtered = [t for t in raw_tokens if t not in self.STOP_WORDS]
        tokens = filtered if filtered else raw_tokens
        counts = Counter(tokens)

        indices = []
        values = []
        for token, count in counts.items():
            idf_val = self.idf.get(token, 1.0)
            indices.append(self.hash_token(token))
            values.append(float(idf_val * count))

        return models.SparseVector(indices=indices, values=values)
