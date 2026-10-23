from src.parser import InsuranceDocParser, DocumentChunk

def test_parser_creates_breadcrumbed_chunks():
    parser = InsuranceDocParser(chunk_size=300, chunk_overlap=50)
    sample_markdown_pages = [
        (1, "# TRAVEL MEDICAL INSURANCE\n## Section 1 - Summary of Benefits\nUp to $5,000,000 emergency medical coverage for travelers under 65."),
        (2, "## Section 6 - Limitations and Exclusions\nPre-existing medical conditions are excluded unless stable for 90 days prior to departure.")
    ]
    chunks = parser._process_markdown_pages(sample_markdown_pages, card_name="TD Aeroplan VIP")

    assert len(chunks) >= 2
    assert all(isinstance(c, DocumentChunk) for c in chunks)

    medical_chunk = [c for c in chunks if "Pre-existing" in c.content][0]
    assert "[Card: TD Aeroplan VIP]" in medical_chunk.breadcrumb
    assert "TRAVEL MEDICAL INSURANCE" in medical_chunk.breadcrumb
    assert "Limitations and Exclusions" in medical_chunk.breadcrumb
    assert medical_chunk.page_number == 2
