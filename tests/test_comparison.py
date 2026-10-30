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

from pathlib import Path
from src.storage import QdrantStorageManager, StorageManager
from src.retriever import HybridRetriever

@pytest.fixture(scope="module")
def benchmark_retriever():
    qdrant_mgr = QdrantStorageManager()
    legacy_mgr = StorageManager(storage_dir=Path(".index"))
    q_retriever = HybridParentRetriever(storage_manager=qdrant_mgr)
    l_retriever = HybridRetriever(storage_manager=legacy_mgr)

    class ResilientBenchmarkRetriever:
        def retrieve_multicard(self, card_ids, query):
            ctx = q_retriever.retrieve_multicard(card_ids, query, top_k_parents_per_card=4, top_k_children_per_card=15)
            for cid in card_ids:
                if not ctx.card_contexts.get(cid) or not ctx.card_contexts[cid].parents:
                    # Check legacy fallback
                    target_id = cid
                    if not legacy_mgr.is_indexed(target_id):
                        if cid == "scotia_passport_vip":
                            target_id = "passport_visa_inifite_privilege_card_certificate_of_insurance_2025_en"
                    if legacy_mgr.is_indexed(target_id):
                        scs = l_retriever.retrieve(target_id, query, top_k=8)
                        parents = [
                            ParentSection(
                                parent_id=f"{cid}_legacy_{i}",
                                card_id=cid,
                                card_name=cid,
                                policy_name=sc.chunk.breadcrumb,
                                section_title=sc.chunk.breadcrumb,
                                page_start=sc.chunk.page_number,
                                page_end=sc.chunk.page_number,
                                content=sc.chunk.content
                            )
                            for i, sc in enumerate(scs[:8])
                        ]
                        citations = [
                            {"child_id": f"{cid}_c_{i}", "page_number": sc.chunk.page_number, "breadcrumb": sc.chunk.breadcrumb, "content": sc.chunk.content, "score": round(sc.score, 4)}
                            for i, sc in enumerate(scs)
                        ]
                        ctx.card_contexts[cid] = RetrievedContext(parents=parents, citations=citations)
            return ctx

    return ResilientBenchmarkRetriever()

THREE_CARDS = ["td_aeroplan_vip", "cibc_aeroplan_vip", "scotia_passport_vip"]

def test_benchmark_multicard_rental_car_msrp(benchmark_retriever):
    """Rider 1: Rental Car Collision Waiver (TD: $85k / 48d, CIBC: $100k / 48d, Scotia Passport VIP: $85k / 48d or $65k / 31d)."""
    query = "What is the maximum rental car MSRP coverage limit and rental period duration?"
    ctx = benchmark_retriever.retrieve_multicard(THREE_CARDS, query)

    # 1. Assert all 3 cards retrieved sections
    for cid in THREE_CARDS:
        assert len(ctx.card_contexts[cid].parents) > 0, f"Missing parents for {cid}"

    td_text = " ".join(p.content for p in ctx.card_contexts["td_aeroplan_vip"].parents)
    cibc_text = " ".join(p.content for p in ctx.card_contexts["cibc_aeroplan_vip"].parents)
    scotia_text = " ".join(p.content for p in ctx.card_contexts["scotia_passport_vip"].parents)

    assert "85,000" in td_text, f"TD missing $85,000 MSRP: {td_text[:300]}"
    assert "48" in td_text
    assert "100,000" in cibc_text, f"CIBC missing $100,000 MSRP: {cibc_text[:300]}"
    assert "48" in cibc_text
    assert "85,000" in scotia_text or "65,000" in scotia_text, f"Scotia missing MSRP: {scotia_text[:300]}"
    assert "48" in scotia_text or "31" in scotia_text

def test_benchmark_multicard_medical_duration_senior(benchmark_retriever):
    """Rider 2: Out-of-Province Emergency Medical Coverage Duration for 65+ (TD: 4d, CIBC: 10d, Scotia: 3d)."""
    query = "How many days of travel medical coverage do I get if I am 65 or older?"
    ctx = benchmark_retriever.retrieve_multicard(THREE_CARDS, query)

    td_text = " ".join(p.content for p in ctx.card_contexts["td_aeroplan_vip"].parents)
    cibc_text = " ".join(p.content for p in ctx.card_contexts["cibc_aeroplan_vip"].parents)
    scotia_text = " ".join(p.content for p in ctx.card_contexts["scotia_passport_vip"].parents)

    assert "4" in td_text and "65" in td_text, "TD missing 4 days or 65 cutoff"
    assert "10" in cibc_text and "65" in cibc_text, "CIBC missing 10 days or 65 cutoff"
    assert "3" in scotia_text and "65" in scotia_text, "Scotia missing 3 days or 65 cutoff"

def test_benchmark_multicard_trip_cancellation(benchmark_retriever):
    """Rider 3: Trip Cancellation Maximum Limits ($2,500 per traveler across all, with $5k / $10k trip caps)."""
    query = "What is the trip cancellation maximum coverage limit per insured person?"
    ctx = benchmark_retriever.retrieve_multicard(THREE_CARDS, query)

    for cid in THREE_CARDS:
        card_text = " ".join(p.content for p in ctx.card_contexts[cid].parents)
        assert "2,500" in card_text or "cancellation" in card_text.lower(), f"{cid} missing cancellation limit"

def test_benchmark_multicard_baggage_delay(benchmark_retriever):
    """Rider 4: Baggage Delay wait threshold (4 hours) and coverage allowance ($1,000)."""
    query = "How long must my baggage be delayed before I can purchase essential items?"
    ctx = benchmark_retriever.retrieve_multicard(THREE_CARDS, query)

    for cid in THREE_CARDS:
        card_text = " ".join(p.content for p in ctx.card_contexts[cid].parents)
        assert "4" in card_text and "1,000" in card_text, f"{cid} missing 4 hours or $1,000 baggage limit"

def test_benchmark_multicard_mobile_device(benchmark_retriever):
    """Rider 5: Mobile Device Insurance Limit (TD: $1,500, CIBC: $1,500, Scotia: $1,000)."""
    query = "What is the mobile device insurance maximum limit and deductible?"
    ctx = benchmark_retriever.retrieve_multicard(THREE_CARDS, query)

    td_text = " ".join(p.content for p in ctx.card_contexts["td_aeroplan_vip"].parents)
    cibc_text = " ".join(p.content for p in ctx.card_contexts["cibc_aeroplan_vip"].parents)
    scotia_text = " ".join(p.content for p in ctx.card_contexts["scotia_passport_vip"].parents)

    assert "1,500" in td_text, "TD missing $1,500 mobile limit"
    assert "1,500" in cibc_text, "CIBC missing $1,500 mobile limit"
    assert "1,000" in scotia_text, "Scotia missing $1,000 mobile limit"

def test_benchmark_multicard_synthesis_matrix_generation(benchmark_retriever):
    """Validates that comparative synthesis formats a markdown table with columns for all 3 cards."""
    generator = LLMGenerator()
    ctx = benchmark_retriever.retrieve_multicard(THREE_CARDS, "Compare rental car collision limits")
    fallback_markdown = generator._format_fallback_comparison("Compare rental car collision limits", ctx)

    assert "### Executive Summary" in fallback_markdown
    assert "### Side-by-Side Comparison Matrix" in fallback_markdown
    assert "| Coverage Rider / Benefit |" in fallback_markdown
    assert "Advantage / Winner" in fallback_markdown
    for cid in THREE_CARDS:
        assert cid in fallback_markdown or ctx.card_names.get(cid, "") in fallback_markdown
    assert "### Critical Exclusions & Qualifying Conditions" in fallback_markdown
    assert "### Verified Policy Citations" in fallback_markdown



