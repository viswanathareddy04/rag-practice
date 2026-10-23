from pathlib import Path
from src.parser import DocumentChunk
from src.storage import StorageManager
from src.retriever import HybridRetriever

def test_hybrid_retriever_ranks_exact_and_semantic(tmp_path: Path):
    storage = StorageManager(storage_dir=tmp_path)
    chunks = [
        DocumentChunk(
            chunk_id="c1",
            card_name="TD VIP",
            breadcrumb="[Card: TD VIP] > [Policy: Flight Delay] > [Section: Coverage] (Page 33)",
            page_number=33,
            content="Flight delay covers up to $1,000 for emergency accommodation after 4 hours delay.",
            full_text_with_context="[Card: TD VIP] > [Policy: Flight Delay] > [Section: Coverage] (Page 33)\n---\nFlight delay covers up to $1,000 for emergency accommodation after 4 hours delay."
        ),
        DocumentChunk(
            chunk_id="c2",
            card_name="TD VIP",
            breadcrumb="[Card: TD VIP] > [Policy: Lost Baggage] > [Section: Coverage] (Page 29)",
            page_number=29,
            content="Lost luggage compensation up to $1,000.",
            full_text_with_context="[Card: TD VIP] > [Policy: Lost Baggage] > [Section: Coverage] (Page 29)\n---\nLost luggage compensation up to $1,000."
        )
    ]
    storage.save_card_index("td_vip", chunks)

    retriever = HybridRetriever(storage_manager=storage)
    results = retriever.retrieve(card_id="td_vip", query="flight delayed by 5 hours hotel room", top_k=1)

    assert len(results) == 1
    assert results[0].chunk.chunk_id == "c1"
    assert results[0].chunk.page_number == 33
    assert results[0].retrieval_method == "hybrid_rrf"
