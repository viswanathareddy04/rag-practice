# Phase 4: Full-Stack Web Application & Multi-Bank Verification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a production-grade FastAPI web service with Server-Sent Events (SSE) streaming, a modern dark fintech Web UI, and verify cross-bank generalization across TD, CIBC, and Scotiabank certificates.

**Architecture:** A unified FastAPI application in `api/main.py` provides REST & SSE endpoints while mounting the static `frontend/` directory. The frontend is built in Vanilla HTML5/CSS/JS with a dark fintech aesthetic and connects to `/api/query` via readable streams. Multi-bank test suite in `tests/test_multibank.py` validates bank agnosticism.

**Tech Stack:** FastAPI, Uvicorn, httpx, Vanilla HTML5 / Modern CSS / ES6 JavaScript, Pytest.

**Spec:** [`docs/specs/phase_4_fullstack_web_and_multibank_design.md`](file:///d:/rag-practice/docs/specs/phase_4_fullstack_web_and_multibank_design.md)

## Global Constraints
- Unified FastAPI server serving both API routes and static frontend on `http://localhost:8000`.
- Vanilla CSS with CSS custom variables for the Dark Fintech theme (No Tailwind).
- Zero placeholders (`TODO`, `TBD`) in production code.
- Strict TDD (write failing test first, verify failure, implement minimal code, verify pass, commit).

---

### Task 1: FastAPI Backend Service & SSE Streaming Endpoints

**Files:**
- Create: `api/__init__.py`
- Create: `api/main.py`
- Create: `tests/test_api.py`

**Interfaces:**
- Consumes:
  - `CardRegistry.scan()` from `src/registry.py`
  - `StorageManager` from `src/storage.py`
  - `HybridRetriever.retrieve()` from `src/retriever.py`
  - `LLMGenerator.generate_stream()` and `extract_citations()` from `src/generator.py`
- Produces:
  - `GET /api/cards`: Returns list of card metadata dicts with `is_indexed` boolean.
  - `POST /api/query`: SSE `EventSource` stream emitting `context`, `token`, `citations`, `done`.
  - `POST /api/index/{card_id}`: On-demand card indexing.
  - `GET /api/status`: System health.

- [ ] **Step 1: Write the failing API unit tests**

```python
# tests/test_api.py
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_api_status():
    response = client.get("/api/status")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "model" in data

def test_api_get_cards():
    response = client.get("/api/cards")
    assert response.status_code == 200
    cards = response.json()
    assert isinstance(cards, list)
    assert len(cards) >= 1
    td_card = next((c for c in cards if "td_aeroplan_vip" in c["card_id"]), None)
    assert td_card is not None
    assert td_card["is_indexed"] is True

def test_api_query_stream():
    payload = {
        "card_id": "td_aeroplan_vip",
        "query": "What is the flight delay coverage limit?"
    }
    with client.stream("POST", "/api/query", json=payload) as response:
        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]
        events = []
        for line in response.iter_lines():
            if line:
                events.append(line)
        assert any("event: context" in e or "event: token" in e for e in events)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_api.py -v`  
Expected: FAIL with `ModuleNotFoundError: No module named 'api'`

- [ ] **Step 3: Implement minimal FastAPI application in `api/main.py`**

```python
# api/main.py
import json
import os
import sys
import time
from pathlib import Path
from typing import AsyncGenerator
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

class QueryRequest(BaseModel):
    card_id: str
    query: str

@app.get("/api/status")
def get_status():
    return {
        "status": "ok",
        "model": generator.model,
        "ollama_base_url": generator.base_url,
        "indexed_cards": [p.name for p in index_dir.iterdir() if p.is_dir()] if index_dir.exists() else []
    }

@app.get("/api/cards")
def get_cards():
    cards = registry.scan()
    results = []
    for c in cards:
        indexed = storage.is_indexed(c.card_id)
        chunks_count = 0
        if indexed:
            try:
                data = storage.load_card_index(c.card_id)
                chunks_count = len(data.chunks)
            except Exception:
                pass
        results.append({
            "card_id": c.card_id,
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
    if not card:
        all_cards = registry.scan()
        matches = [c for c in all_cards if card_id.lower() in c.card_id]
        if matches:
            card = matches[0]
            card_id = card.card_id
        else:
            raise HTTPException(status_code=404, detail=f"Card '{card_id}' not found")

    start = time.time()
    parser = InsuranceDocParser(chunk_size=400, chunk_overlap=80)
    chunks = parser.parse_pdf(card.file_path, card_name=card.card_name, card_id=card.card_id)
    storage.save_card_index(card.card_id, chunks)
    return {
        "status": "success",
        "card_id": card.card_id,
        "chunks_indexed": len(chunks),
        "elapsed_seconds": round(time.time() - start, 2)
    }

async def sse_event_generator(card_id: str, query: str):
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
    if not storage.is_indexed(req.card_id):
        index_card(req.card_id)
    return StreamingResponse(
        sse_event_generator(req.card_id, req.query),
        media_type="text/event-stream"
    )

if frontend_dir.exists():
    app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_api.py -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add api/ tests/test_api.py ; git commit -m "feat(api): implement FastAPI server with SSE streaming query and card endpoints"
```

---

### Task 2: Sleek Dark Fintech Web UI (HTML, CSS, JS)

**Files:**
- Create: `frontend/index.html`
- Create: `frontend/style.css`
- Create: `frontend/app.js`

**Interfaces:**
- Consumes:
  - `GET /api/cards`
  - `POST /api/query` (SSE Stream)
  - `POST /api/index/{card_id}`
- Produces:
  - Full-featured browser UI at `http://localhost:8000`.

- [ ] **Step 1: Create `frontend/index.html`**
  - Semantic structure: Header with Shield logo, Card selector dropdown, Indexing status pill.
  - Split grid: Left chat panel with message thread and query input; Right collapsible drawer for citations.
  - Suggested query chips.

- [ ] **Step 2: Create `frontend/style.css`**
  - Dark Fintech aesthetic (`#080C14` base, `#0E1526` surface, `#131C31` card).
  - Glassmorphic panels with subtle borders (`rgba(255, 255, 255, 0.08)`).
  - Typography: Google Fonts `Inter` + `JetBrains Mono`.
  - Micro-animations, responsive layout.

- [ ] **Step 3: Create `frontend/app.js`**
  - Card discovery on load (`fetchCards()`).
  - Native `ReadableStream` reader parsing SSE events (`context`, `token`, `citations`, `done`).
  - Markdown formatting for `###`, bullet points, bolding, and tables.
  - Expandable citation drawers.

- [ ] **Step 4: Verify static mount and frontend files**

Run: `python -c "import httpx; assert Path('frontend/index.html').exists()"`  
Run: `pytest tests/test_api.py -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add frontend/ ; git commit -m "feat(ui): implement modern dark fintech Web UI with SSE chat and citations drawer"
```

---

### Task 3: Cross-Bank & Multi-Card Automated Test Suite

**Files:**
- Create: `tests/test_multibank.py`

**Interfaces:**
- Consumes:
  - `CardRegistry` scanning `data/` (`cibc/`, `scotia/`, `td/`)
  - `InsuranceDocParser.parse_pdf()`
  - `StorageManager`
  - `HybridRetriever.retrieve()`

- [ ] **Step 1: Write `tests/test_multibank.py`**

```python
# tests/test_multibank.py
from pathlib import Path
from src.registry import CardRegistry
from src.parser import InsuranceDocParser
from src.storage import StorageManager
from src.retriever import HybridRetriever

def test_registry_discovers_all_three_banks():
    registry = CardRegistry(base_dir=Path("data"))
    cards = registry.scan()
    banks = set(c.bank for c in cards)
    assert "TD" in banks
    assert "CIBC" in banks
    assert "SCOTIA" in banks
    assert len(cards) >= 6

def test_cibc_card_ingestion_and_retrieval(tmp_path):
    cibc_pdf = Path("data/cibc/CIBC Aeroplan Visa Infinite Privilege Card-Insurance.pdf")
    assert cibc_pdf.exists()

    parser = InsuranceDocParser(chunk_size=400, chunk_overlap=80)
    chunks = parser.parse_pdf(str(cibc_pdf), card_name="CIBC Aeroplan VIP", extract_dir=None)
    assert len(chunks) > 30
    assert any("[Card: CIBC Aeroplan VIP]" in c.breadcrumb for c in chunks)

    storage = StorageManager(storage_dir=tmp_path)
    storage.save_card_index("cibc_aeroplan_vip", chunks)

    retriever = HybridRetriever(storage_manager=storage)
    results = retriever.retrieve("cibc_aeroplan_vip", "flight delay reimbursement", top_k=5)
    assert len(results) > 0
    assert results[0].chunk.page_number >= 1

def test_scotia_card_ingestion_and_retrieval(tmp_path):
    scotia_pdf = Path("data/scotia/Passport_Visa_Inifite_Privilege_Card_Certificate_of_Insurance_2025_En.pdf")
    assert scotia_pdf.exists()

    parser = InsuranceDocParser(chunk_size=400, chunk_overlap=80)
    chunks = parser.parse_pdf(str(scotia_pdf), card_name="Scotiabank Passport VIP", extract_dir=None)
    assert len(chunks) > 30
    assert any("[Card: Scotiabank Passport VIP]" in c.breadcrumb for c in chunks)

    storage = StorageManager(storage_dir=tmp_path)
    storage.save_card_index("scotia_passport_vip", chunks)

    retriever = HybridRetriever(storage_manager=storage)
    results = retriever.retrieve("scotia_passport_vip", "rental vehicle collision damage", top_k=5)
    assert len(results) > 0
```

- [ ] **Step 2: Run test to verify it passes**

Run: `pytest tests/test_multibank.py -v`  
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_multibank.py ; git commit -m "feat(multibank): add cross-bank ingestion and retrieval validation for CIBC and Scotiabank"
```

---

### Task 4: End-to-End Verification & Phase 4 Sign-Off

**Files:**
- Modify: `docs/ROADMAP.md`
- Modify: `docs/phases/PHASE_4_FULLSTACK_UI.md`

- [ ] **Step 1: Run full test suite**

Run: `pytest -v`  
Expected: 23+ passed tests across unit, benchmark, API, and multi-bank suites.

- [ ] **Step 2: Launch server and verify HTTP response**

Run: `uvicorn api.main:app --port 8000` (background or quick curl check).  
Verify `GET http://localhost:8000/api/cards` returns 200 OK.  
Verify `GET http://localhost:8000/` returns 200 OK with HTML content.

- [ ] **Step 3: Update documentation and checklists**
  - Mark Phase 4 as **DONE (100%) ✅** in `docs/ROADMAP.md` and `docs/phases/PHASE_4_FULLSTACK_UI.md`.

- [ ] **Step 4: Commit**

```bash
git add docs/ ; git commit -m "docs: mark Phase 4 fullstack web UI and multi-bank verification complete"
```
