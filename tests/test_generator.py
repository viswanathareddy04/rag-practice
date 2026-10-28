from unittest.mock import MagicMock
from src.parser import DocumentChunk, ParentSection
from src.retriever import ScoredChunk, RetrievedContext
from src.generator import LLMGenerator, GroundedResponse

def test_generator_builds_parent_context_prompt():
    gen = LLMGenerator()
    parent = ParentSection(
        parent_id="p1", card_id="cibc_vip", card_name="CIBC VIP",
        policy_name="Flight Delay", section_title="Benefits", page_start=35, page_end=36,
        content="Flight delay pays up to $500 for hotel and food after 4 hours."
    )
    citation = {
        "page_number": 36, "breadcrumb": "[CIBC] > [Flight Delay] (Page 36)",
        "content": "Flight delay pays up to $500."
    }
    context = RetrievedContext(parents=[parent], citations=[citation])

    prompt = gen._build_context_prompt("How much do I get for delayed flight?", context)
    assert "--- Parent Section 1: [Flight Delay] > Benefits (Pages 35-36) ---" in prompt
    assert "Flight delay pays up to $500 for hotel and food" in prompt
    assert "USER QUESTION:" in prompt


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

def test_generator_streams_tokens():
    mock_client = MagicMock()
    # Client stream returns iterator of token dicts
    mock_client.chat.return_value = [
        {"message": {"content": "### Direct "}},
        {"message": {"content": "Answer\n"}},
        {"message": {"content": "Coverage is confirmed."}}
    ]

    generator = LLMGenerator(ollama_client=mock_client)
    sample_chunk = DocumentChunk(
        chunk_id="c1",
        card_name="TD VIP",
        breadcrumb="[Card: TD VIP] > [Policy: Flight Delay] > [Section: Section 5] (Page 33)",
        page_number=33,
        content="Flight delay covers up to $1,000.",
        full_text_with_context="[Card: TD VIP] > [Policy: Flight Delay] > [Section: Section 5] (Page 33)\n---\nFlight delay covers up to $1,000."
    )
    scored = [ScoredChunk(chunk=sample_chunk, score=0.9, retrieval_method="hybrid")]

    tokens = list(generator.generate_stream("Am I covered?", scored))
    full_output = "".join(tokens)
    assert "### Direct Answer" in full_output
    assert "Coverage is confirmed." in full_output

def test_generator_fallback_when_offline():
    # Generator with non-existent base URL to trigger fallback
    generator = LLMGenerator(base_url="http://localhost:99999")
    sample_chunk = DocumentChunk(
        chunk_id="c1",
        card_name="TD VIP",
        breadcrumb="[Card: TD VIP] > [Policy: Flight Delay] > [Section: Section 5] (Page 33)",
        page_number=33,
        content="Flight delay covers up to $1,000.",
        full_text_with_context="[Card: TD VIP] > [Policy: Flight Delay] > [Section: Section 5] (Page 33)\n---\nFlight delay covers up to $1,000."
    )
    scored = [ScoredChunk(chunk=sample_chunk, score=0.9, retrieval_method="hybrid")]

    # Normal generate fallback
    resp = generator.generate("What is the flight delay limit?", scored)
    assert "### Direct Answer" in resp.answer_markdown
    assert "Flight delay covers up to $1,000." in resp.answer_markdown
    assert len(resp.citations) == 1

    # Streaming generate fallback
    stream_tokens = list(generator.generate_stream("What is the flight delay limit?", scored))
    stream_output = "".join(stream_tokens)
    assert "### Direct Answer" in stream_output
    assert "Flight delay covers up to $1,000." in stream_output

