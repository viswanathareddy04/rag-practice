import pytest
from pathlib import Path
from src.registry import CardRegistry, CardMetadata

def test_card_registry_discovers_cards(tmp_path: Path):
    # Setup mock bank directories and dummy pdf files
    td_dir = tmp_path / "td"
    td_dir.mkdir()
    td_pdf = td_dir / "TD Aeroplan Visa Infinite Privilege Card COI.pdf"
    td_pdf.write_bytes(b"%PDF-dummy")

    cibc_dir = tmp_path / "cibc"
    cibc_dir.mkdir()
    cibc_pdf = cibc_dir / "CIBC Dividend Visa Infinite Card-Insurance.pdf"
    cibc_pdf.write_bytes(b"%PDF-dummy")

    registry = CardRegistry(base_dir=tmp_path)
    cards = registry.scan()

    assert len(cards) == 2
    card_ids = [c.card_id for c in cards]
    assert "td_aeroplan_visa_infinite_privilege_card_coi" in card_ids
    assert "cibc_dividend_visa_infinite_card_insurance" in card_ids

    td_card = registry.get_card("td_aeroplan_visa_infinite_privilege_card_coi")
    assert td_card is not None
    assert td_card.bank.lower() == "td"
    assert td_card.file_path == str(td_pdf.resolve())
    assert "TD Aeroplan Visa Infinite Privilege" in td_card.card_name
