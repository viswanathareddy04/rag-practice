from pathlib import Path
import pytest
from qdrant_client import QdrantClient
from src.parser import InsuranceDocParser
from src.storage import QdrantStorageManager
from src.retriever import HybridParentRetriever

@pytest.fixture(scope="module")
def indexed_cibc_card():
    # In-memory client for fast testing
    client = QdrantClient(location=":memory:")
    storage = QdrantStorageManager(client=client, model_name="sentence-transformers/all-MiniLM-L6-v2", device="cpu")

    pdf_path = Path("data/cibc/CIBC Aeroplan Visa Infinite Privilege Card-Insurance.pdf")
    assert pdf_path.exists(), f"CIBC PDF not found at {pdf_path}"

    parser = InsuranceDocParser(chunk_size=400, chunk_overlap=80)
    parents, children = parser.parse_pdf_parent_child(
        str(pdf_path),
        card_name="CIBC Aeroplan Visa Infinite Privilege Card",
        card_id="cibc_aeroplan_vip",
        extract_dir=None
    )
    assert len(parents) >= 10, f"Expected at least 10 parent sections, got {len(parents)}"
    assert len(children) >= 30, f"Expected at least 30 child chunks, got {len(children)}"

    storage.save_card_chunks("cibc_aeroplan_vip", parents, children)
    return HybridParentRetriever(storage_manager=storage)

def test_cibc_benchmark_rental_car_msrp(indexed_cibc_card):
    retriever = indexed_cibc_card
    ctx = retriever.retrieve("cibc_aeroplan_vip", "What is the maximum rental car MSRP coverage limit and rental period?")
    assert len(ctx.parents) > 0
    assert len(ctx.citations) > 0

    combined_parent_text = "\n".join(p.content for p in ctx.parents)
    assert "100,000" in combined_parent_text
    assert "48" in combined_parent_text

def test_cibc_benchmark_medical_duration_by_age(indexed_cibc_card):
    retriever = indexed_cibc_card
    ctx = retriever.retrieve("cibc_aeroplan_vip", "How many days of travel medical coverage do I get if I am 65 or older?")
    assert len(ctx.parents) > 0

    combined_parent_text = "\n".join(p.content for p in ctx.parents)
    assert "10" in combined_parent_text
    assert "65" in combined_parent_text
    assert "31" in combined_parent_text

def test_cibc_benchmark_negative_pet_coverage(indexed_cibc_card):
    retriever = indexed_cibc_card
    ctx = retriever.retrieve("cibc_aeroplan_vip", "Does this policy cover veterinary medical emergencies for dogs?")
    assert len(ctx.parents) > 0
    # Pet coverage should not exist in medical benefits
    combined_parent_text = "\n".join(p.content.lower() for p in ctx.parents)
    assert "dog" not in combined_parent_text
    assert "veterinary" not in combined_parent_text
