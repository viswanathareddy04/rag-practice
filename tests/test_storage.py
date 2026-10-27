from pathlib import Path
from src.parser import DocumentChunk
from src.storage import StorageManager

def test_storage_manager_persists_and_loads(tmp_path: Path):
    storage = StorageManager(storage_dir=tmp_path)
    sample_chunks = [
        DocumentChunk(
            chunk_id="c1",
            card_name="TD VIP",
            breadcrumb="[Card: TD VIP] > [Policy: Medical] (Page 2)",
            page_number=2,
            content="Medical coverage up to $5M.",
            full_text_with_context="[Card: TD VIP] > [Policy: Medical] (Page 2)\n---\nMedical coverage up to $5M."
        )
    ]
    storage.save_card_index("td_vip", sample_chunks)
    assert storage.is_indexed("td_vip") is True

    loaded_index = storage.load_card_index("td_vip")
    assert len(loaded_index.chunks) == 1
    assert loaded_index.chunks[0].chunk_id == "c1"
    assert loaded_index.bm25 is not None
    assert loaded_index.embeddings is not None
    assert loaded_index.embeddings.shape[0] == 1
