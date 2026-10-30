import json
import os
import sys
import time
from pathlib import Path
from typing import Generator, List
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.registry import CardRegistry
from src.parser import InsuranceDocParser, ParentSection
from src.storage import StorageManager, QdrantStorageManager
from src.retriever import HybridRetriever, HybridParentRetriever, RetrievedContext
from src.generator import LLMGenerator

app = FastAPI(title="Credit Card Insurance RAG API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

data_dir = project_root / "data"
index_dir = project_root / ".index"
frontend_dir = project_root / "frontend"

registry = CardRegistry(base_dir=data_dir)
# Local fallback storage manager
legacy_storage = StorageManager(storage_dir=index_dir)
legacy_retriever = HybridRetriever(storage_manager=legacy_storage, rrf_k=60)

# Cloud Qdrant storage manager
qdrant_storage = QdrantStorageManager()
qdrant_retriever = HybridParentRetriever(storage_manager=qdrant_storage, rrf_k=60)

generator = LLMGenerator(model=os.getenv("LLM_MODEL", "qwen2.5:7b"))

CARD_ALIASES = {
    "td_aeroplan_visa_infinite_privilege_card_coi": "td_aeroplan_vip",
    "cibc_aeroplan_visa_infinite_privilege_card_insurance": "cibc_aeroplan_vip",
    "passport_visa_inifite_privilege_card_certificate_of_insurance_2025_en": "scotia_passport_vip",
}
REVERSE_ALIASES = {v: k for k, v in CARD_ALIASES.items()}

class QueryRequest(BaseModel):
    card_id: str
    query: str

@app.get("/api/status")
def get_status():
    indexed_list = [p.name for p in index_dir.iterdir() if p.is_dir()] if index_dir.exists() else []
    return {
        "status": "ok",
        "model": generator.model,
        "ollama_base_url": generator.base_url,
        "qdrant_collection": qdrant_storage.COLLECTION_NAME,
        "indexed_cards": indexed_list
    }

@app.get("/api/cards")
def get_cards():
    cards = registry.scan()
    results = []
    for c in cards:
        effective_id = CARD_ALIASES.get(c.card_id, c.card_id)
        indexed = qdrant_storage.is_indexed(effective_id) or legacy_storage.is_indexed(effective_id) or legacy_storage.is_indexed(c.card_id)
        chunks_count = 0
        if legacy_storage.is_indexed(effective_id) or legacy_storage.is_indexed(c.card_id):
            target_id = effective_id if legacy_storage.is_indexed(effective_id) else c.card_id
            try:
                data = legacy_storage.load_card_index(target_id)
                chunks_count = len(data.chunks)
            except Exception:
                pass
        results.append({
            "card_id": effective_id,
            "raw_id": c.card_id,
            "bank": c.bank,
            "card_name": c.card_name,
            "file_path": c.file_path,
            "is_indexed": indexed,
            "total_chunks": chunks_count
        })
    return results

@app.post("/api/index/{card_id}")
def index_card(card_id: str):
    card = registry.get_card(card_id)
    if not card and card_id in REVERSE_ALIASES:
        card = registry.get_card(REVERSE_ALIASES[card_id])
    if not card:
        all_cards = registry.scan()
        matches = [
            c for c in all_cards
            if card_id.lower() in c.card_id or card_id.lower() in CARD_ALIASES.get(c.card_id, "").lower()
        ]
        if matches:
            card = matches[0]
        else:
            raise HTTPException(status_code=404, detail=f"Card '{card_id}' not found")

    start = time.time()
    parser = InsuranceDocParser(chunk_size=400, chunk_overlap=80)
    effective_id = CARD_ALIASES.get(card.card_id, card.card_id)

    # 1. Parse into ParentSections and ChildChunks
    parents, children = parser.parse_pdf_parent_child(card.file_path, card_name=card.card_name, card_id=effective_id)
    # 2. Index into Qdrant Cloud
    qdrant_storage.save_card_chunks(effective_id, parents, children)

    # Also build legacy index for offline safety
    legacy_chunks = parser.parse_pdf(card.file_path, card_name=card.card_name, card_id=effective_id)
    legacy_storage.save_card_index(effective_id, legacy_chunks)
    if effective_id != card.card_id:
        legacy_storage.save_card_index(card.card_id, legacy_chunks)

    return {
        "status": "success",
        "card_id": effective_id,
        "parent_sections": len(parents),
        "children_indexed": len(children),
        "elapsed_seconds": round(time.time() - start, 2)
    }

def sse_event_generator(card_id: str, query: str) -> Generator[str, None, None]:
    start_total = time.time()

    # Prefer Qdrant Cloud Parent-Document Retriever
    if qdrant_storage.is_indexed(card_id):
        retrieved_ctx = qdrant_retriever.retrieve(card_id=card_id, query=query, top_k_parents=3, top_k_children=8)
        
        context_payload = [
            {
                "rank": i + 1,
                "page_number": c.get("page_number", 1),
                "breadcrumb": c.get("breadcrumb", ""),
                "snippet": c.get("content", "")[:200] + "..." if len(c.get("content", "")) > 200 else c.get("content", ""),
                "full_content": c.get("content", ""),
                "score": round(c.get("score", 0.0), 4)
            }
            for i, c in enumerate(retrieved_ctx.citations)
        ]
        yield f"event: context\ndata: {json.dumps(context_payload)}\n\n"

        tokens_stream = generator.generate_stream(query, retrieved_ctx)
        for token in tokens_stream:
            yield f"event: token\ndata: {json.dumps(token)}\n\n"

        yield f"event: citations\ndata: {json.dumps(retrieved_ctx.citations)}\n\n"

    else:
        # Fallback to legacy local retriever
        scored_chunks = legacy_retriever.retrieve(card_id=card_id, query=query, top_k=8)

        context_payload = [
            {
                "rank": i + 1,
                "page_number": sc.chunk.page_number,
                "breadcrumb": sc.chunk.breadcrumb,
                "snippet": sc.chunk.content[:200] + "..." if len(sc.chunk.content) > 200 else sc.chunk.content,
                "full_content": sc.chunk.content,
                "score": round(sc.score, 4)
            }
            for i, sc in enumerate(scored_chunks)
        ]
        yield f"event: context\ndata: {json.dumps(context_payload)}\n\n"

        tokens_stream = generator.generate_stream(query, scored_chunks)
        for token in tokens_stream:
            yield f"event: token\ndata: {json.dumps(token)}\n\n"

        citations_payload = generator.extract_citations(scored_chunks)
        yield f"event: citations\ndata: {json.dumps(citations_payload)}\n\n"

    elapsed = round((time.time() - start_total) * 1000, 1)
    yield f"event: done\ndata: {json.dumps({'elapsed_ms': elapsed})}\n\n"

@app.post("/api/query")
def stream_query(req: QueryRequest):
    card_id = req.card_id
    if not (qdrant_storage.is_indexed(card_id) or legacy_storage.is_indexed(card_id)):
        alt_id = REVERSE_ALIASES.get(card_id, CARD_ALIASES.get(card_id))
        if alt_id and (qdrant_storage.is_indexed(alt_id) or legacy_storage.is_indexed(alt_id)):
            card_id = alt_id
        else:
            index_card(card_id)
            if not (qdrant_storage.is_indexed(card_id) or legacy_storage.is_indexed(card_id)):
                card_id = CARD_ALIASES.get(card_id, card_id)

    return StreamingResponse(
        sse_event_generator(card_id, req.query),
        media_type="text/event-stream"
    )

class CompareRequest(BaseModel):
    card_ids: List[str]
    query: str

def sse_compare_event_generator(card_ids: List[str], query: str) -> Generator[str, None, None]:
    start_total = time.time()

    # Multi-card retrieval
    multi_ctx = qdrant_retriever.retrieve_multicard(
        card_ids=card_ids,
        query=query,
        top_k_parents_per_card=4,
        top_k_children_per_card=15
    )

    # Fallback to legacy index for any card that yielded empty parents
    for cid in card_ids:
        ctx = multi_ctx.card_contexts.get(cid)
        if (not ctx or not ctx.parents) and (legacy_storage.is_indexed(cid) or (cid in REVERSE_ALIASES and legacy_storage.is_indexed(REVERSE_ALIASES[cid]))):
            target_id = cid if legacy_storage.is_indexed(cid) else REVERSE_ALIASES[cid]
            legacy_chunks = legacy_retriever.retrieve(card_id=target_id, query=query, top_k=8)
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
                    for i, sc in enumerate(legacy_chunks[:6])
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

    # 1. Emit cards_context event

    context_payload = {
        cid: [
            {
                "rank": i + 1,
                "page_number": c.get("page_number", 1),
                "breadcrumb": c.get("breadcrumb", ""),
                "snippet": c.get("content", "")[:200] + "..." if len(c.get("content", "")) > 200 else c.get("content", ""),
                "full_content": c.get("content", ""),
                "score": round(c.get("score", 0.0), 4)
            }
            for i, c in enumerate(ctx.citations)
        ]
        for cid, ctx in multi_ctx.card_contexts.items()
    }
    yield f"event: cards_context\ndata: {json.dumps(context_payload)}\n\n"

    # 2. Stream tokens
    token_stream = generator.generate_comparison_stream(query, multi_ctx)
    for token in token_stream:
        yield f"event: token\ndata: {json.dumps(token)}\n\n"

    # 3. Emit grouped citations
    citations_payload = {
        cid: ctx.citations
        for cid, ctx in multi_ctx.card_contexts.items()
    }
    yield f"event: citations\ndata: {json.dumps(citations_payload)}\n\n"

    elapsed = round((time.time() - start_total) * 1000, 1)
    yield f"event: done\ndata: {json.dumps({'elapsed_ms': elapsed})}\n\n"

@app.post("/api/compare")
def stream_compare(req: CompareRequest):
    if len(req.card_ids) < 2 or len(req.card_ids) > 3:
        raise HTTPException(status_code=400, detail="Comparison requires between 2 and 3 cards.")

    resolved_ids = []
    for cid in req.card_ids:
        eff_id = CARD_ALIASES.get(cid, cid)
        if not (qdrant_storage.is_indexed(eff_id) or legacy_storage.is_indexed(eff_id)):
            index_card(eff_id)
            if not (qdrant_storage.is_indexed(eff_id) or legacy_storage.is_indexed(eff_id)):
                eff_id = CARD_ALIASES.get(eff_id, eff_id)
        resolved_ids.append(eff_id)

    return StreamingResponse(
        sse_compare_event_generator(resolved_ids, req.query),
        media_type="text/event-stream"
    )

if frontend_dir.exists():
    app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")
