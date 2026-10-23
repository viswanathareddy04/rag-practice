import pytest
from pathlib import Path
from src.parser import InsuranceDocParser
from src.storage import StorageManager
from src.retriever import HybridRetriever

@pytest.fixture(scope="module")
def indexed_td_card(tmp_path_factory):
    storage_dir = tmp_path_factory.mktemp("td_index")
    pdf_path = Path("d:/rag-practice/data/td/TD Aeroplan Visa Infinite Privilege Card COI.pdf")
    assert pdf_path.exists(), f"TD PDF not found at {pdf_path}"

    parser = InsuranceDocParser(chunk_size=400, chunk_overlap=80)
    chunks = parser.parse_pdf(str(pdf_path), card_name="TD Aeroplan Visa Infinite Privilege Card")
    assert len(chunks) > 40, f"Expected at least 40 chunks for 52-page TD COI, got {len(chunks)}"

    storage = StorageManager(storage_dir=storage_dir)
    storage.save_card_index("td_aeroplan_vip", chunks)
    return HybridRetriever(storage_manager=storage)

BENCHMARK_CASES = [
    # 1. Flight Delay threshold and allowance (Page 33)
    ("What is the minimum delay time and maximum coverage for flight delay?", 33, ["1,000", "4 hours"]),
    # 2. Medical Pre-existing stability (Pages 7-8)
    ("What are the pre-existing medical condition stability requirements?", [7, 8], ["stable", "pre-existing"]),
    # 3. Rental car MSRP cap (Page 38: Section 5 Types of Vehicles Covered)
    ("What is the maximum MSRP for rental car collision damage coverage?", 38, ["85,000", "msrp"]),
    # 4. Baggage delay threshold (Page 29)
    ("How long must my baggage be delayed before I can purchase essential items?", 29, ["4 hours", "1,000"]),
    # 5. Mobile device insurance limit & deductible (Page 46)
    ("What is the mobile device insurance maximum limit and deductible?", 46, ["1,500", "10%"]),
    # 6. Medical age cutoff (Pages 1, 3, 4, 10)
    ("How many days of travel medical coverage do I get if I am 65 or older?", [1, 3, 4, 10], ["4 consecutive days", "65"]),
    # 7. Trip Cancellation valid cause vs voluntary (Pages 16, 18)
    ("Is voluntary change of travel plans covered under trip cancellation?", [16, 18], ["cancellation"]),
    # 8. Extended Warranty period (Page 40)
    ("How long is the warranty doubled under extended warranty protection?", 40, ["warranty"]),
    # 9. Hotel Burglary coverage limit (Page 50)
    ("What is the limit for hotel or motel burglary coverage?", 50, ["2,500", "burglary"]),
    # 10. Negative uncovered test (General overview / not covered)
    ("Does this insurance certificate cover pet health emergencies during travel?", None, [])
]

@pytest.mark.parametrize("query,expected_page,key_terms", BENCHMARK_CASES)
def test_td_insurance_benchmarks(indexed_td_card, query, expected_page, key_terms):
    retriever = indexed_td_card
    results = retriever.retrieve(card_id="td_aeroplan_vip", query=query, top_k=5)
    assert len(results) > 0

    retrieved_pages = [r.chunk.page_number for r in results]
    retrieved_texts = " ".join([r.chunk.content.lower() for r in results])

    if expected_page is not None:
        target_pages = expected_page if isinstance(expected_page, list) else [expected_page]
        matched_any = any(p in retrieved_pages for p in target_pages)
        assert matched_any, f"Query '{query}' failed: expected pages in {target_pages}, got {retrieved_pages}"

        for term in key_terms:
            assert term.lower() in retrieved_texts, f"Query '{query}' missing term '{term}' in retrieved context: {retrieved_texts[:300]}"
