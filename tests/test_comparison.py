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
