# Credit Card Insurance RAG: Phased Roadmap & Live Status

**Last Updated:** 2026-10-23  
**Target Baseline Card:** TD Aeroplan Visa Infinite Privilege Card (52 pages, 10 policies)  
**Data Directory:** `d:\rag-practice\data\` (Strictly dedicated to raw input PDFs: `cibc/`, `scotia/`, `td/`)  
**Project Root:** `d:\rag-practice\` (`src/`, `tests/`, `docs/`, `.index/`)  

---

## 1. Phase Status Overview

| Phase | Focus Area | Deliverables | Current Status | Detailed Guide |
| :--- | :--- | :--- | :--- | :--- |
| **Setup & Cleanup** | Directory Organization | Keep `data/` clean for raw PDFs only; move code to root | **DONE** ✅ | — |
| **Phase 1** | Ingestion & Breadcrumb Parsing | Dynamic card discovery, TOC/layout parser, breadcrumbs | **DONE (100%)** ✅ | [Phase 1 Guide](phases/PHASE_1_INGESTION.md) |
| **Phase 2** | Hybrid Retrieval & Benchmark | Card-scoped BM25Plus + Dense RRF store, 10/10 tests pass | **DONE (100%)** ✅ | [Phase 2 Guide](phases/PHASE_2_RETRIEVAL.md) |
| **Phase 3** | Grounded Synthesis & Guardrails | Local Ollama LLM integration, zero-hallucination prompt | **DONE (100%)** ✅ | [Phase 3 Guide](phases/PHASE_3_SYNTHESIS.md) |
| **Phase 4** | Web Interface & Multi-Bank | FastAPI service, Vanilla HTML/CSS/JS UI, Scotia/CIBC validation | **DONE (100%)** ✅ | [Phase 4 Guide](phases/PHASE_4_FULLSTACK_UI.md) |

> **Architectural Changes & History**: See [Design Change Records](design_changes/README.md) (e.g. [DCR-001: Phase 1 Parser Refactor](design_changes/001_phase1_parser_hierarchical_tracking.md)).

---

## 2. Detailed Status by Phase

### Phase 1: Ingestion & Hierarchical Breadcrumb Parsing
* **Goal:** Ingest any bank's digital insurance PDF, preserve markdown tables and headings, and prepend hierarchical breadcrumbs (`[Card] > [Policy] > [Section] (Page X)`) without bank-specific deterministic regex.
* **Status:** **COMPLETE (100%) ✅**
  - Dynamic card discovery in `src/registry.py`.
  - Layout-aware parser in `src/parser.py` with continuous markdown export.

---

### Phase 2: Hybrid Retrieval & Ground Truth Benchmark
* **Goal:** Deliver reliable, millisecond search combining lexical BM25 (for exact numbers, limits, and policy terms) and dense semantic vectors (`sentence-transformers/all-MiniLM-L6-v2`) via Reciprocal Rank Fusion (RRF).
* **Status:** **COMPLETE (100%) ✅**
  - Card-scoped storage in `src/storage.py` (`.index/<card_id>/`).
  - Dual-index RRF retriever in `src/retriever.py` with question stopword filtering.
  - 10/10 benchmark test cases passing with 100% precision (`tests/test_benchmark_td.py`).

---

### Phase 3: Grounded Synthesis with Citation Guardrails
* **Goal:** Connect local Ollama (`qwen2.5:7b`) with a strict grounding prompt and 4-part structured response schema with exact page citations.
* **Status:** **COMPLETE (100%) ✅**
  - Streaming generator `src/generator.py` with offline fallback.
  - Interactive terminal client `scripts/query.py`.
  - 5/5 automated synthesis benchmark evaluation cases passing with 100% schema compliance and negative rejection (`scripts/eval_synthesis.py`).

---

### Phase 4: Full-Stack Web UI & Cross-Bank Verification
* **Goal:** FastAPI backend + modern Dark Fintech Web UI (card dropdown, streaming chat, collapsible source inspector) and verifying end-to-end functionality across TD, CIBC, and Scotiabank certificates.
* **Status:** **COMPLETE (100%) ✅**
  - FastAPI server in `api/main.py` serving `/api/status`, `/api/cards`, `/api/query` (SSE streaming tokens and structured citations), `/api/index/{card_id}`, and mounting `frontend/` on `http://localhost:8000/`.
  - Frontend UI in Vanilla HTML5, CSS, and ES6 JS (`frontend/index.html`, `style.css`, `app.js`) with Dark Fintech aesthetics (`#080C14`), glassmorphism, quick prompts, real-time token streaming, and collapsible citations drawer.
  - Multi-bank test suite in `tests/test_multibank.py` verifying dynamic discovery, ingestion, and hybrid retrieval across CIBC and Scotiabank certificates.
  - Full automated test suite passing (27/27 tests in `pytest -v`).

