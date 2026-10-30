import argparse
import sys
import time
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.storage import QdrantStorageManager, StorageManager
from src.retriever import HybridParentRetriever, HybridRetriever, RetrievedContext
from src.generator import LLMGenerator
from src.parser import ParentSection

CARD_ALIASES = {
    "td_aeroplan_visa_infinite_privilege_card_coi": "td_aeroplan_vip",
    "cibc_aeroplan_visa_infinite_privilege_card_insurance": "cibc_aeroplan_vip",
    "passport_visa_inifite_privilege_card_certificate_of_insurance_2025_en": "scotia_passport_vip",
}
REVERSE_ALIASES = {v: k for k, v in CARD_ALIASES.items()}

def main():
    parser = argparse.ArgumentParser(description="Multi-Card Insurance Policy Comparison Terminal Client")
    parser.add_argument("--cards", nargs="+", required=True, help="2 or 3 Card IDs to compare (e.g. td_aeroplan_vip cibc_aeroplan_vip)")
    parser.add_argument("--query", type=str, required=True, help="Comparison query (e.g. 'Compare rental car collision limits')")
    args = parser.parse_args()

    if len(args.cards) < 2 or len(args.cards) > 3:
        print("\033[91mError: Please specify 2 or 3 cards to compare.\033[0m")
        sys.exit(1)

    print(f"\n\033[96m========================================================================\033[0m")
    print(f"\033[96m  CREDIT CARD INSURANCE RAG — MULTI-CARD COMPARISON ENGINE\033[0m")
    print(f"\033[96m  Cards: {', '.join(args.cards)}\033[0m")
    print(f"\033[96m  Query: {args.query}\033[0m")
    print(f"\033[96m========================================================================\033[0m\n")

    storage = QdrantStorageManager()
    legacy_storage = StorageManager(storage_dir=project_root / ".index")
    legacy_retriever = HybridRetriever(storage_manager=legacy_storage)
    retriever = HybridParentRetriever(storage_manager=storage)
    generator = LLMGenerator()

    start_retrieval = time.time()
    print("\033[93m[1/2] Fan-out hybrid retrieval across card booklets...\033[0m")
    multi_ctx = retriever.retrieve_multicard(card_ids=args.cards, query=args.query)

    # Offline / disk fallback check if any card had 0 parents
    for cid in args.cards:
        ctx = multi_ctx.card_contexts.get(cid)
        if (not ctx or not ctx.parents) and (legacy_storage.is_indexed(cid) or (cid in REVERSE_ALIASES and legacy_storage.is_indexed(REVERSE_ALIASES[cid]))):
            target_id = cid if legacy_storage.is_indexed(cid) else REVERSE_ALIASES[cid]
            legacy_chunks = legacy_retriever.retrieve(card_id=target_id, query=args.query, top_k=6)
            if legacy_chunks:
                parents = [
                    ParentSection(
                        parent_id=f"{cid}_legacy_{i}",
                        card_id=cid,
                        card_name=multi_ctx.card_names.get(cid, cid),
                        policy_name=sc.chunk.breadcrumb.split(" > ")[1] if " > " in sc.chunk.breadcrumb else "General Coverage",
                        section_title=sc.chunk.breadcrumb.split(" > ")[-1] if " > " in sc.chunk.breadcrumb else "Benefits",
                        page_start=sc.chunk.page_number,
                        page_end=sc.chunk.page_number,
                        content=sc.chunk.content
                    )
                    for i, sc in enumerate(legacy_chunks[:2])
                ]
                citations = [
                    {
                        "child_id": f"{cid}_c_{i}",
                        "page_number": sc.chunk.page_number,
                        "breadcrumb": sc.chunk.breadcrumb,
                        "content": sc.chunk.content,
                        "score": round(sc.score, 4)
                    }
                    for i, sc in enumerate(legacy_chunks)
                ]
                multi_ctx.card_contexts[cid] = RetrievedContext(parents=parents, citations=citations)

    retrieval_ms = round((time.time() - start_retrieval) * 1000, 1)

    for cid, ctx in multi_ctx.card_contexts.items():
        name = multi_ctx.card_names.get(cid, cid)
        print(f"  • \033[92m{name}\033[0m: {len(ctx.parents)} parent sections, {len(ctx.citations)} citations")

    print(f"\n\033[93m[2/2] Streaming comparative synthesis (retrieval: {retrieval_ms}ms)...\033[0m\n")
    start_gen = time.time()
    token_count = 0
    for token in generator.generate_comparison_stream(args.query, multi_ctx):
        sys.stdout.write(token)
        sys.stdout.flush()
        token_count += 1

    gen_s = round(time.time() - start_gen, 2)
    print(f"\n\n\033[90m------------------------------------------------------------------------\033[0m")
    print(f"\033[90mComparison complete in {gen_s}s ({token_count} tokens streamed).\033[0m\n")

if __name__ == "__main__":
    main()
