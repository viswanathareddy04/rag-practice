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

def test_multi_section_same_page_and_subsections():
    parser = InsuranceDocParser(chunk_size=300, chunk_overlap=50)
    sample_pages = [
        (2, "# Section 1 - Summary of Benefits\nUp to $5,000,000 emergency medical coverage for travelers under 65.\n\n# Section 2 - Introduction\nThis Certificate of Insurance applies to TD Credit Cards.")
    ]
    # Simulate TOC with policy mapping
    toc = [
        (1, "TRAVEL MEDICAL INSURANCE", 1),
        (2, "Section 1 - Summary of Benefits", 2),
        (2, "Section 2 - Introduction", 2),
        (2, "Section 3 - Eligibility", 2)
    ]
    policy_map, section_map = parser._build_toc_maps(toc, total_pages=5)
    chunks = parser._process_markdown_pages(sample_pages, card_name="TD Aeroplan VIP", 
                                            toc_policy=policy_map, toc_section=section_map, toc=toc)

    assert len(chunks) == 2
    sec1_chunk = [c for c in chunks if "Summary of Benefits" in c.content][0]
    sec2_chunk = [c for c in chunks if "Certificate of Insurance" in c.content][0]
    assert "Section 1 - Summary of Benefits" in sec1_chunk.breadcrumb
    assert "Section 2 - Introduction" in sec2_chunk.breadcrumb

def test_policy_boundary_does_not_inherit_previous_policy_section():
    parser = InsuranceDocParser(chunk_size=300, chunk_overlap=50)
    sample_pages = [
        (11, "Section 10 General Conditions terms for travel medical.\n" * 3),
        (12, "Welcome to Trip Cancellation insurance. Important terms apply.\n" * 3)
    ]
    toc = [
        (1, "TRAVEL MEDICAL INSURANCE", 1),
        (2, "Section 10 - General Conditions", 11),
        (1, "TRIP CANCELLATION / TRIP INTERRUPTION INSURANCE", 12),
        (2, "Section 1 - Introduction", 13)
    ]
    policy_map, section_map = parser._build_toc_maps(toc, total_pages=15)
    chunks = parser._process_markdown_pages(sample_pages, card_name="TD Aeroplan VIP", 
                                            toc_policy=policy_map, toc_section=section_map, toc=toc)

    p12_chunks = [c for c in chunks if c.page_number == 12]
    assert len(p12_chunks) > 0
    for c in p12_chunks:
        assert "TRIP CANCELLATION" in c.breadcrumb
        assert "General Conditions" not in c.breadcrumb

def test_save_extracted_markdown_to_disk(tmp_path):
    parser = InsuranceDocParser()
    sample_pages = [
        (1, "# TRAVEL MEDICAL INSURANCE\n## Section 1 - Summary of Benefits\nUp to $5,000,000 coverage."),
        (2, "## Section 2 - Introduction\nTerms and conditions.")
    ]
    out_file = parser._save_extracted_markdown(
        pages_md=sample_pages,
        card_name="TD Aeroplan VIP",
        card_id="td_aeroplan_vip",
        output_dir=str(tmp_path)
    )
    assert out_file.exists()
    assert out_file.name == "td_aeroplan_vip.md"
    content = out_file.read_text(encoding="utf-8")
    assert "# TD Aeroplan VIP" in content
    assert "TRAVEL MEDICAL INSURANCE" in content
    assert "<!-- Page 1 -->" in content
    assert "<!-- Page 2 -->" in content
