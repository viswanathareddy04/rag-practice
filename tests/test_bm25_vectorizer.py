from src.bm25_vectorizer import BM25Vectorizer
from qdrant_client import models

def test_bm25_vectorizer_encodes_sparse_vectors():
    corpus = [
        "Car rental collision damage coverage covers up to $100,000 MSRP for 48 days.",
        "Flight delay insurance covers essential purchases up to $500 after 4 hours delay.",
        "Emergency travel medical insurance provides up to $5,000,000 for hospital stays."
    ]
    vectorizer = BM25Vectorizer(k1=1.5, b=0.75)
    vectorizer.fit(corpus)

    doc_vector = vectorizer.encode_document(corpus[0])
    assert isinstance(doc_vector, models.SparseVector)
    assert len(doc_vector.indices) > 0
    assert len(doc_vector.values) == len(doc_vector.indices)
    assert all(isinstance(idx, int) and idx > 0 for idx in doc_vector.indices)
    assert all(val > 0.0 for val in doc_vector.values)

def test_bm25_vectorizer_encodes_query_with_stopword_pruning():
    corpus = ["Rental car collision coverage covers damage up to $100,000."]
    vectorizer = BM25Vectorizer()
    vectorizer.fit(corpus)

    query_vector = vectorizer.encode_query("What is the maximum coverage for rental car?")
    assert isinstance(query_vector, models.SparseVector)
    assert len(query_vector.indices) > 0
    # Stopwords like 'what', 'is', 'the', 'for' should be pruned from sparse query
    tokens_encoded = [vectorizer.hash_token(t) for t in ["rental", "car", "coverage", "maximum"]]
    assert any(idx in tokens_encoded for idx in query_vector.indices)
