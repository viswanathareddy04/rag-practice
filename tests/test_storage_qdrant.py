from qdrant_client import QdrantClient
from src.parser import ParentSection, ChildChunk
from src.storage import QdrantStorageManager

def test_qdrant_storage_manager_upserts_and_indexes():
    client = QdrantClient(location=":memory:")
    # Use mini model or mock for fast unit test
    storage = QdrantStorageManager(client=client, model_name="sentence-transformers/all-MiniLM-L6-v2", device="cpu")

    parent = ParentSection(
        parent_id="cibc_vip__med__sec8",
        card_id="cibc_aeroplan_vip",
        card_name="CIBC Aeroplan VIP",
        policy_name="TRAVEL MEDICAL INSURANCE",
        section_title="SECTION 8 - WHAT ARE YOU NOT COVERED FOR?",
        page_start=7,
        page_end=9,
        content="Full text of Section 8..."
    )
    child = ChildChunk(
        child_id="cibc_vip__med__sec8__c001",
        parent_id=parent.parent_id,
        card_id=parent.card_id,
        card_name=parent.card_name,
        policy_name=parent.policy_name,
        section_title=parent.section_title,
        page_number=8,
        content="Pre-existing medical conditions not stable within 90 days are excluded.",
        breadcrumb="[Card: CIBC Aeroplan VIP] > [Medical] (Page 8)",
        parent_content=parent.content
    )

    storage.save_card_chunks(card_id="cibc_aeroplan_vip", parent_sections=[parent], child_chunks=[child])

    assert storage.is_indexed("cibc_aeroplan_vip")
    info = client.get_collection(storage.COLLECTION_NAME)
    assert info.points_count == 1

    points, _ = client.scroll(collection_name=storage.COLLECTION_NAME, limit=1)
    p = points[0]
    assert p.payload["card_id"] == "cibc_aeroplan_vip"
    assert p.payload["parent_content"] == parent.content
    assert p.payload["page_number"] == 8
    assert p.payload["parent_id"] == parent.parent_id
