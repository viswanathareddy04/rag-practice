#!/usr/bin/env python3
"""
CLI script to index all credit card insurance booklets into Qdrant Cloud 
using the Parent-Document Retrieval architecture.
"""

import argparse
import os
import sys
import time
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Load .env if present
env_file = PROJECT_ROOT / ".env"
if env_file.exists():
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, val = line.split("=", 1)
            os.environ.setdefault(key.strip(), val.strip())

from src.registry import CardRegistry
from src.parser import InsuranceDocParser
from src.storage import QdrantStorageManager, StorageManager

def index_cards(card_filter: str = None):
    print("=" * 60)
    print(" Credit Card Insurance RAG — Cloud Ingestion Engine")
    print("=" * 60)

    data_dir = PROJECT_ROOT / "data"
    registry = CardRegistry(base_dir=data_dir)
    cards = registry.scan()

    if not cards:
        print("[!] No PDF booklets found in data/ directory.")
        return

    storage = QdrantStorageManager()
    legacy_storage = StorageManager(storage_dir=PROJECT_ROOT / ".index")
    parser = InsuranceDocParser(chunk_size=400, chunk_overlap=80)

    print(f"[*] Connecting to Qdrant...")
    print(f"    Collection: {storage.COLLECTION_NAME}")
    print(f"    Embedding Model: {storage.model_name} ({storage.device})")
    storage.create_collection_if_not_exists()

    if card_filter:
        cards = [c for c in cards if card_filter.lower() in c.card_id.lower() or card_filter.lower() in c.card_name.lower()]
        if not cards:
            print(f"[!] No card matched filter: '{card_filter}'")
            return

    print(f"[*] Found {len(cards)} card(s) to index.\n")
    total_parents = 0
    total_children = 0
    total_start = time.time()

    for idx, card in enumerate(cards, 1):
        print(f"[{idx}/{len(cards)}] Processing: {card.card_name} ({card.bank.upper()})")
        print(f"    File: {card.file_path}")
        start = time.time()

        try:
            # 1. Parse into ParentSections and ChildChunks
            parents, children = parser.parse_pdf_parent_child(
                card.file_path, card_name=card.card_name, card_id=card.card_id
            )
            print(f"    Extracted: {len(parents)} Parent Sections | {len(children)} Child Chunks")

            # 2. Upload to Qdrant Cloud
            storage.save_card_chunks(card.card_id, parents, children)
            print(f"    Uploaded to Qdrant Cloud in {time.time() - start:.2f}s")

            # 3. Save legacy index for fallback
            legacy_chunks = parser.parse_pdf(card.file_path, card_name=card.card_name, card_id=card.card_id)
            legacy_storage.save_card_index(card.card_id, legacy_chunks)

            total_parents += len(parents)
            total_children += len(children)
            print(f"    Status: SUCCESS [OK]\n")

        except Exception as e:
            print(f"    Status: FAILED with error: {e}\n")

    elapsed = time.time() - total_start
    print("=" * 60)
    print(f"[+] All Done in {elapsed:.2f}s!")
    print(f"    Total Parent Sections: {total_parents}")
    print(f"    Total Child Vectors:   {total_children}")
    print("=" * 60)

if __name__ == "__main__":
    cli_parser = argparse.ArgumentParser(description="Index credit card booklets into Qdrant Cloud.")
    cli_parser.add_argument("--card", type=str, default=None, help="Specific card_id or name filter to index.")
    args = cli_parser.parse_args()

    index_cards(card_filter=args.card)
