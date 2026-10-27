from pathlib import Path
from src.registry import CardRegistry
from src.parser import InsuranceDocParser
from src.storage import StorageManager
from src.retriever import HybridRetriever

def test_registry_discovers_all_three_banks():
    registry = CardRegistry(base_dir=Path("data"))
    cards = registry.scan()
    banks = set(c.bank for c in cards)
    assert "TD" in banks
    assert "CIBC" in banks
    assert "SCOTIA" in banks
    assert len(cards) >= 6

def test_cibc_card_ingestion_and_retrieval(tmp_path):
    cibc_pdf = Path("data/cibc/CIBC Aeroplan Visa Infinite Privilege Card-Insurance.pdf")
    assert cibc_pdf.exists()

    parser = InsuranceDocParser(chunk_size=400, chunk_overlap=80)
    chunks = parser.parse_pdf(str(cibc_pdf), card_name="CIBC Aeroplan VIP", extract_dir=None)
    assert len(chunks) > 30
    assert any("[Card: CIBC Aeroplan VIP]" in c.breadcrumb for c in chunks)

    storage = StorageManager(storage_dir=tmp_path)
    storage.save_card_index("cibc_aeroplan_vip", chunks)

    retriever = HybridRetriever(storage_manager=storage)
    results = retriever.retrieve("cibc_aeroplan_vip", "flight delay reimbursement", top_k=5)
    assert len(results) > 0
    assert results[0].chunk.page_number >= 1

def test_scotia_card_ingestion_and_retrieval(tmp_path):
    scotia_pdf = Path("data/scotia/Passport_Visa_Inifite_Privilege_Card_Certificate_of_Insurance_2025_En.pdf")
    assert scotia_pdf.exists()

    parser = InsuranceDocParser(chunk_size=400, chunk_overlap=80)
    chunks = parser.parse_pdf(str(scotia_pdf), card_name="Scotiabank Passport VIP", extract_dir=None)
    assert len(chunks) > 30
    assert any("[Card: Scotiabank Passport VIP]" in c.breadcrumb for c in chunks)

    storage = StorageManager(storage_dir=tmp_path)
    storage.save_card_index("scotia_passport_vip", chunks)

    retriever = HybridRetriever(storage_manager=storage)
    results = retriever.retrieve("scotia_passport_vip", "rental vehicle collision damage", top_k=5)
    assert len(results) > 0
    assert results[0].chunk.page_number >= 1
