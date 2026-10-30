from unittest.mock import MagicMock
import pytest
from src.parser import ParentSection
from src.retriever import HybridParentRetriever, RetrievedContext, MultiCardContext

def test_retriever_multicard_parallel_fanout():
    mock_storage = MagicMock()
    retriever = HybridParentRetriever(storage_manager=mock_storage)

    # Mock retrieve method to return card-specific parents
    def fake_retrieve(card_id: str, query: str, top_k_parents: int = 2, top_k_children: int = 6):
        return RetrievedContext(
            parents=[ParentSection(
                parent_id=f"{card_id}_p1",
                card_id=card_id,
                card_name=f"Card Name for {card_id}",
                policy_name="Flight Delay",
                section_title="Benefits",
                page_start=10,
                page_end=11,
                content=f"Coverage content for {card_id}"
            )],
            citations=[{"child_id": f"{card_id}_c1", "page_number": 10, "breadcrumb": "Flight Delay", "content": "Excerpt", "score": 0.85}]
        )

    retriever.retrieve = MagicMock(side_effect=fake_retrieve)

    card_ids = ["td_aeroplan_vip", "cibc_aeroplan_vip", "scotia_passport_vip"]
    result = retriever.retrieve_multicard(card_ids=card_ids, query="What is the flight delay coverage limit?")

    assert isinstance(result, MultiCardContext)
    assert len(result.card_contexts) == 3
    assert len(result.card_names) == 3
    for cid in card_ids:
        assert cid in result.card_contexts
        ctx = result.card_contexts[cid]
        assert len(ctx.parents) == 1
        assert ctx.parents[0].card_id == cid
        assert cid in result.card_names

from src.generator import LLMGenerator, COMPARISON_SYSTEM_PROMPT

def test_generator_comparison_prompt_assembly():
    generator = LLMGenerator()
    multi_ctx = MultiCardContext(
        card_contexts={
            "card_a": RetrievedContext(
                parents=[ParentSection("p1", "card_a", "Bank A Card", "Rental Car", "Limits", 10, 10, "MSRP up to $85,000 for 48 days.")],
                citations=[]
            ),
            "card_b": RetrievedContext(
                parents=[ParentSection("p2", "card_b", "Bank B Card", "Rental Car", "Limits", 20, 20, "MSRP up to $100,000 for 48 days.")],
                citations=[]
            )
        },
        card_names={"card_a": "Bank A Card", "card_b": "Bank B Card"}
    )

    prompt = generator._build_multicard_context_prompt("Compare rental car limits", multi_ctx)
    assert "=== CARD: Bank A Card (ID: card_a) ===" in prompt
    assert "=== CARD: Bank B Card (ID: card_b) ===" in prompt
    assert "MSRP up to $85,000" in prompt
    assert "MSRP up to $100,000" in prompt
    assert "COMPARATIVE QUESTION:\nCompare rental car limits" in prompt

def test_generator_comparison_fallback_markdown():
    generator = LLMGenerator()
    multi_ctx = MultiCardContext(
        card_contexts={
            "td_aeroplan_vip": RetrievedContext(
                parents=[ParentSection("p1", "td_aeroplan_vip", "TD Aeroplan VIP", "Rental Car", "Limits", 38, 38, "Maximum MSRP $85,000.")],
                citations=[]
            ),
            "cibc_aeroplan_vip": RetrievedContext(
                parents=[ParentSection("p2", "cibc_aeroplan_vip", "CIBC Aeroplan VIP", "Rental Car", "Limits", 24, 24, "Maximum MSRP $100,000.")],
                citations=[]
            )
        },
        card_names={"td_aeroplan_vip": "TD Aeroplan VIP", "cibc_aeroplan_vip": "CIBC Aeroplan VIP"}
    )

    fallback = generator._format_fallback_comparison("Compare rental car limits", multi_ctx)
    assert "### Executive Summary" in fallback
    assert "### Side-by-Side Comparison Matrix" in fallback
    assert "| Coverage Rider / Benefit |" in fallback
    assert "TD Aeroplan VIP" in fallback
    assert "CIBC Aeroplan VIP" in fallback
    assert "### Critical Exclusions & Qualifying Conditions" in fallback
    assert "### Verified Policy Citations" in fallback

from fastapi.testclient import TestClient
from api.main import app

api_client = TestClient(app)

def test_api_compare_validation_requires_2_or_3_cards():
    # Only 1 card
    res1 = api_client.post("/api/compare", json={"card_ids": ["td_aeroplan_vip"], "query": "Compare limits"})
    assert res1.status_code == 400
    assert "between 2 and 3 cards" in res1.json()["detail"]

    # 4 cards
    res2 = api_client.post("/api/compare", json={
        "card_ids": ["card_1", "card_2", "card_3", "card_4"],
        "query": "Compare limits"
    })
    assert res2.status_code == 400

def test_api_compare_streaming_events():
    payload = {
        "card_ids": ["td_aeroplan_vip", "cibc_aeroplan_vip"],
        "query": "Compare rental car collision limits"
    }
    with api_client.stream("POST", "/api/compare", json=payload) as response:
        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]
        events = []
        for line in response.iter_lines():
            if line:
                events.append(line)
        assert any("event: cards_context" in e for e in events)
        assert any("event: token" in e for e in events)
        assert any("event: citations" in e for e in events)
        assert any("event: done" in e for e in events)


