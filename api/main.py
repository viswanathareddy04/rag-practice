import json
import os
import sys
import time
from pathlib import Path
from typing import Generator
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.registry import CardRegistry
from src.parser import InsuranceDocParser
from src.storage import StorageManager
from src.retriever import HybridRetriever
from src.generator import LLMGenerator

app = FastAPI(title="Credit Card Insurance RAG API", version="1.0.0")

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
storage = StorageManager(storage_dir=index_dir)
retriever = HybridRetriever(storage_manager=storage, rrf_k=60)
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
        "indexed_cards": indexed_list
    }

@app.get("/api/cards")
def get_cards():
    cards = registry.scan()
    results = []
    for c in cards:
        effective_id = CARD_ALIASES.get(c.card_id, c.card_id)
        indexed = storage.is_indexed(effective_id) or storage.is_indexed(c.card_id)
        chunks_count = 0
        if indexed:
            target_id = effective_id if storage.is_indexed(effective_id) else c.card_id
            try:
                data = storage.load_card_index(target_id)
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
    chunks = parser.parse_pdf(card.file_path, card_name=card.card_name, card_id=effective_id)
    storage.save_card_index(effective_id, chunks)
    if effective_id != card.card_id:
        storage.save_card_index(card.card_id, chunks)

    return {
        "status": "success",
        "card_id": effective_id,
        "chunks_indexed": len(chunks),
        "elapsed_seconds": round(time.time() - start, 2)
    }

def sse_event_generator(card_id: str, query: str) -> Generator[str, None, None]:
    start_total = time.time()
    scored_chunks = retriever.retrieve(card_id=card_id, query=query, top_k=8)

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
    if not storage.is_indexed(card_id):
        alt_id = REVERSE_ALIASES.get(card_id, CARD_ALIASES.get(card_id))
        if alt_id and storage.is_indexed(alt_id):
            card_id = alt_id
        else:
            index_card(card_id)
            if not storage.is_indexed(card_id):
                card_id = CARD_ALIASES.get(card_id, card_id)

    return StreamingResponse(
        sse_event_generator(card_id, req.query),
        media_type="text/event-stream"
    )

if frontend_dir.exists():
    app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")
