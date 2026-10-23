from unittest.mock import MagicMock
from src.parser import DocumentChunk
from src.retriever import ScoredChunk
from src.generator import LLMGenerator, GroundedResponse

def test_generator_formats_prompt_and_parses_citations():
    mock_client = MagicMock()
    mock_client.chat.return_value = {
        "message": {
            "content": "### Direct Answer\nYes, flight delays are covered.\n\n"
                       "### Coverage Limits & Details\nUp to $1,000 for 4+ hour delays.\n\n"
                       "### Key Conditions & Exclusions\nMust charge full ticket to card.\n\n"
                       "### Policy Citations\n[Flight Delay Insurance, Section 5, Page 33]"
        }
    }

    generator = LLMGenerator(ollama_client=mock_client)
    sample_chunk = DocumentChunk(
        chunk_id="c1",
        card_name="TD VIP",
        breadcrumb="[Card: TD VIP] > [Policy: Flight Delay] > [Section: Section 5] (Page 33)",
        page_number=33,
        content="Flight delay covers up to $1,000 for emergency accommodation after 4 hours delay.",
        full_text_with_context="[Card: TD VIP] > [Policy: Flight Delay] > [Section: Section 5] (Page 33)\n---\nFlight delay covers up to $1,000."
    )
    scored = [ScoredChunk(chunk=sample_chunk, score=0.9, retrieval_method="hybrid")]

    response = generator.generate("Am I covered for flight delay?", scored)
    assert isinstance(response, GroundedResponse)
    assert "Yes, flight delays are covered" in response.answer_markdown
    assert len(response.citations) == 1
    assert response.citations[0]["page_number"] == 33
    assert response.citations[0]["breadcrumb"] == sample_chunk.breadcrumb
