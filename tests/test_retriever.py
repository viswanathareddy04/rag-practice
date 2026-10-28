from pathlib import Path
from qdrant_client import QdrantClient
from src.parser import DocumentChunk, ParentSection, ChildChunk
from src.storage import StorageManager, QdrantStorageManager
from src.retriever import HybridRetriever, HybridParentRetriever, RetrievedContext

def test_retriever_hybrid_rrf_and_parent_deduplication():
    client = QdrantClient(location=":memory:")
    storage = QdrantStorageManager(client=client, model_name="sentence-transformers/all-MiniLM-L6-v2", device="cpu")

    p1 = ParentSection(
        parent_id="td__car_rental__benefits", card_id="td_vip", card_name="TD VIP",
        policy_name="Car Rental Insurance", section_title="Benefits", page_start=24, page_end=25,
        content="Coverage covers rental cars up to $85,000 MSRP for up to 48 days."
    )
    p2 = ParentSection(
        parent_id="td__car_rental__exclusions", card_id="td_vip", card_name="TD VIP",
        policy_name="Car Rental Insurance", section_title="Exclusions", page_start=26, page_end=27,
        content="Excludes commercial vans, off-road vehicles, and exotic cars."
    )

    c1 = ChildChunk("td__c1", p1.parent_id, "td_vip", "TD VIP", "Car Rental", "Benefits", 24,
                    "MSRP limit is $85,000 for rental vehicles.", "[TD] (Page 24)", p1.content)
    c2 = ChildChunk("td__c2", p1.parent_id, "td_vip", "TD VIP", "Car Rental", "Benefits", 25,
                    "Rental duration limit is 48 consecutive days.", "[TD] (Page 25)", p1.content)
    c3 = ChildChunk("td__c3", p2.parent_id, "td_vip", "TD VIP", "Car Rental", "Exclusions", 26,
                    "Commercial vans and off-road driving are excluded.", "[TD] (Page 26)", p2.content)

    storage.save_card_chunks("td_vip", [p1, p2], [c1, c2, c3])

    retriever = HybridParentRetriever(storage_manager=storage)
    result = retriever.retrieve(card_id="td_vip", query="What is the maximum car rental MSRP?")

    assert isinstance(result, RetrievedContext)
    parent_ids = [p.parent_id for p in result.parents]
    assert len(parent_ids) == len(set(parent_ids))
    assert p1.parent_id in parent_ids

    assert len(result.citations) >= 1
    assert result.citations[0]["page_number"] in [24, 25]
    assert "85,000" in result.citations[0]["content"]

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
