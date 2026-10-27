#!/usr/bin/env python3
"""
Interactive Credit Card Insurance Q&A Terminal Client.
Supports live streaming generation with local Ollama and structured citation inspection.
"""

import argparse
import sys
import time
from pathlib import Path

# Reconfigure stdout for UTF-8 compatibility on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Add project root to sys.path so 'src' can be imported when run as script
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.registry import CardRegistry
from src.parser import InsuranceDocParser
from src.storage import StorageManager
from src.retriever import HybridRetriever
from src.generator import LLMGenerator

# ANSI Terminal Colors
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
MAGENTA = "\033[95m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"

def print_banner():
    print(f"\n{BOLD}{CYAN}======================================================================{RESET}")
    print(f"{BOLD}{CYAN}      CREDIT CARD INSURANCE RAG - GROUNDED Q&A TERMINAL ASSISTANT     {RESET}")
    print(f"{BOLD}{CYAN}======================================================================{RESET}")
    print(f"{DIM}Powered by Hybrid RRF (BM25Plus + MiniLM-L6-v2) & Local Qwen 2.5 7B{RESET}\n")

def ensure_card_indexed(card_id: str, registry: CardRegistry, storage: StorageManager) -> str:
    """Ensures card is indexed; if not, dynamically parses and indexes it."""
    if storage.is_indexed(card_id):
        return card_id

    card = registry.get_card(card_id)
    if not card:
        print(f"{YELLOW}Warning: Card '{card_id}' not found in registry. Searching by substring...{RESET}")
        all_cards = registry.scan()
        matches = [c for c in all_cards if card_id.lower() in c.card_id or card_id.lower() in c.card_name.lower()]
        if matches:
            card = matches[0]
            card_id = card.card_id
        else:
            raise FileNotFoundError(f"No insurance PDF registered for card ID: '{card_id}'")

    print(f"{YELLOW}* Card '{card.card_name}' is not indexed yet. Indexing now from raw PDF...{RESET}")
    start = time.time()
    parser = InsuranceDocParser(chunk_size=400, chunk_overlap=80)
    chunks = parser.parse_pdf(card.file_path, card_name=card.card_name, card_id=card.card_id)
    storage.save_card_index(card.card_id, chunks)
    print(f"{GREEN}[OK] Successfully indexed {len(chunks)} chunks in {time.time() - start:.1f}s{RESET}\n")
    return card_id

def display_retrieved_context(scored_chunks):
    print(f"\n{BOLD}{BLUE}--- TOP RETRIEVED CERTIFICATE SECTIONS ({len(scored_chunks)} Chunks) ---{RESET}")
    for i, sc in enumerate(scored_chunks):
        c = sc.chunk
        print(f" {DIM}[{i+1}]{RESET} {BOLD}{c.breadcrumb}{RESET} {DIM}(Score: {sc.score:.4f}){RESET}")
    print(f"{BOLD}{BLUE}-------------------------------------------------------------{RESET}\n")

def execute_query(retriever: HybridRetriever, generator: LLMGenerator, card_id: str, query: str):
    print(f"\n{BOLD}{CYAN}QUESTION:{RESET} {BOLD}{query}{RESET}")
    print(f"{DIM}Retrieving grounded excerpts via Hybrid RRF...{RESET}")

    start_ret = time.time()
    scored_chunks = retriever.retrieve(card_id=card_id, query=query, top_k=8)
    ret_time = (time.time() - start_ret) * 1000

    if not scored_chunks:
        print(f"{YELLOW}No matching certificate sections found for this query.{RESET}")
        return

    display_retrieved_context(scored_chunks)

    print(f"{BOLD}{GREEN}### GROUNDED RESPONSE (Model: {generator.model}){RESET}")
    print(f"{DIM}(Retrieval latency: {ret_time:.1f}ms - Streaming answer...){RESET}\n")

    stream_tokens = generator.generate_stream(query, scored_chunks)
    full_text = []
    for token in stream_tokens:
        sys.stdout.write(token)
        sys.stdout.flush()
        full_text.append(token)
    print("\n")

    citations = generator.extract_citations(scored_chunks)
    print(f"{BOLD}{MAGENTA}--- VERIFIED CITATIONS & EXCERPTS ---{RESET}")
    for cit in citations[:4]:
        print(f" {BOLD}* Page {cit['page_number']}:{RESET} {cit['breadcrumb']}")
        print(f"   {DIM}\"{cit['snippet'].strip()}\"{RESET}")
    print(f"{BOLD}{MAGENTA}--------------------------------------{RESET}\n")

def interactive_loop(registry: CardRegistry, storage: StorageManager, retriever: HybridRetriever, generator: LLMGenerator, default_card: str):
    cards = registry.scan()
    print(f"{BOLD}Available Cards in Database:{RESET}")
    for idx, c in enumerate(cards):
        prefix = f"{GREEN}* {RESET}" if c.card_id == default_card else "  "
        print(f"{prefix}[{idx+1}] {BOLD}{c.bank}{RESET} - {c.card_name} {DIM}(ID: {c.card_id}){RESET}")

    current_card = default_card
    current_card = ensure_card_indexed(current_card, registry, storage)

    print(f"\n{GREEN}Active Card set to: {BOLD}{current_card}{RESET}")
    print(f"{DIM}Type your question, or 'switch' to change card, or 'exit' / 'quit' to quit.{RESET}\n")

    while True:
        try:
            user_input = input(f"{BOLD}{CYAN}insurance-rag [{current_card}] > {RESET}").strip()
            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit", "q"):
                print("Goodbye!")
                break
            if user_input.lower() == "switch":
                print("\nSelect a card number to switch:")
                for idx, c in enumerate(cards):
                    print(f"  [{idx+1}] {c.bank} - {c.card_name}")
                choice = input("Enter choice: ").strip()
                if choice.isdigit() and 1 <= int(choice) <= len(cards):
                    new_card = cards[int(choice) - 1].card_id
                    current_card = ensure_card_indexed(new_card, registry, storage)
                    print(f"{GREEN}Switched to: {current_card}{RESET}\n")
                else:
                    print(f"{YELLOW}Invalid choice.{RESET}\n")
                continue

            execute_query(retriever, generator, current_card, user_input)
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

def main():
    parser = argparse.ArgumentParser(description="Credit Card Insurance RAG Interactive Client")
    parser.add_argument("--card", default="td_aeroplan_vip", help="Card ID to query (default: td_aeroplan_vip)")
    parser.add_argument("--query", "-q", default=None, help="One-shot question to answer")
    parser.add_argument("--model", "-m", default="qwen2.5:7b", help="Ollama LLM model name")
    parser.add_argument("--base-url", default="http://localhost:11434", help="Ollama API base URL")
    args = parser.parse_args()

    print_banner()

    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / "data"
    index_dir = project_root / ".index"

    registry = CardRegistry(base_dir=data_dir)
    storage = StorageManager(storage_dir=index_dir)
    retriever = HybridRetriever(storage_manager=storage, rrf_k=60)
    generator = LLMGenerator(model=args.model, base_url=args.base_url)

    card_id = ensure_card_indexed(args.card, registry, storage)

    if args.query:
        execute_query(retriever, generator, card_id, args.query)
    else:
        interactive_loop(registry, storage, retriever, generator, card_id)

if __name__ == "__main__":
    main()
