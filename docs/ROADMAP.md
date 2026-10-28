# Credit Card Insurance RAG: Phased Roadmap & Live Status

**Last Updated:** 2026-10-28  
**Architecture:** Bank-Agnostic Two-Tier Parent-Document Retrieval with Qdrant Cloud Native Hybrid Search  
**Target Baseline Cards:** TD Aeroplan VIP (52 pages), CIBC Aeroplan VIP, Scotiabank Passport VIP  
**Data Directory:** `data/` (Strictly dedicated to raw input PDFs: `cibc/`, `scotia/`, `td/`)  
**Extracted Markdown:** `data/extracted/` (Standardized continuous Markdown per card)  
**Project Root:** `d:\rag-practice\` (`src/`, `tests/`, `docs/`, `api/`, `frontend/`, `scripts/`)  

---

## 1. Phase Status Overview

| Phase | Focus Area | Deliverables | Current Status | Detailed Guide |
| :--- | :--- | :--- | :--- | :--- |
| **Setup & Cleanup** | Directory Organization | Keep `data/` clean for raw PDFs only; isolate code and tests | **DONE** ✅ | — |
| **Phase 1** | Ingestion & Breadcrumb Parsing | Dynamic card discovery, TOC/layout parser, breadcrumbs | **DONE (100%)** ✅ | [Phase 1 Guide](phases/PHASE_1_INGESTION.md) |
| **Phase 2** | Hybrid Retrieval & Benchmark | Card-scoped BM25Plus + Dense RRF store, 10/10 tests pass | **DONE (100%)** ✅ | [Phase 2 Guide](phases/PHASE_2_RETRIEVAL.md) |
| **Phase 3** | Grounded Synthesis & Guardrails | Local Ollama LLM integration, zero-hallucination prompt | **DONE (100%)** ✅ | [Phase 3 Guide](phases/PHASE_3_SYNTHESIS.md) |
| **Phase 4** | Web Interface & Multi-Bank | FastAPI service, Vanilla HTML/CSS/JS UI, Scotia/CIBC validation | **DONE (100%)** ✅ | [Phase 4 Guide](phases/PHASE_4_FULLSTACK_UI.md) |
| **Phase 5** | Parent-Document & Qdrant Cloud | Universal `ParentSection`/`ChildChunk` parser, Cloud BM25 sparse vectors, BGE-base (768-d), server-side RRF fusion | **DONE (100%)** ✅ | [Phase 5 Guide](phases/PHASE_5_PARENT_DOCUMENT_QDRANT.md) |

> **Architectural Changes & History**: See [Design Change Records](design_changes/README.md) (e.g. [DCR-001: Phase 1 Parser Refactor](design_changes/001_phase1_parser_hierarchical_tracking.md), [DCR-002: Markdown Document Export](design_changes/002_markdown_export_on_ingestion.md), [DCR-003: Parent-Document Qdrant Cloud Migration](design_changes/003_parent_document_qdrant_cloud.md)).

---

## 2. Detailed Status by Phase

### Phase 1: Ingestion & Hierarchical Breadcrumb Parsing
* **Goal:** Ingest any bank's digital insurance PDF, preserve markdown tables and headings, and prepend hierarchical breadcrumbs (`[Card] > [Policy] > [Section] (Page X)`) without bank-specific deterministic regex.
* **Status:** **COMPLETE (100%) ✅**
  - Dynamic card discovery in `src/registry.py`.
  - Layout-aware parser in `src/parser.py` with continuous markdown export to `data/extracted/<card_id>.md`.
  - Dual-source outline + layout parsing avoiding cross-page policy bleed.

---

### Phase 2: Hybrid Retrieval & Ground Truth Benchmark
* **Goal:** Deliver reliable search combining lexical BM25 (for exact numbers, limits, and policy terms) and dense semantic vectors (`sentence-transformers/all-MiniLM-L6-v2`) via Reciprocal Rank Fusion (RRF).
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
  - FastAPI server in `api/main.py` serving `/api/status`, `/api/cards`, `/api/query` (SSE streaming tokens and structured citations), `/api/index/{card_id}`, and mounting `frontend/` on `http://localhost:8001/`.
  - Frontend UI in Vanilla HTML5, CSS, and ES6 JS (`frontend/index.html`, `style.css`, `app.js`) with Dark Fintech aesthetics (`#080C14`), glassmorphism, quick prompts, real-time token streaming, and collapsible citations drawer.
  - Multi-bank test suite in `tests/test_multibank.py` verifying dynamic discovery, ingestion, and hybrid retrieval across CIBC and Scotiabank certificates.

---

### Phase 5: Parent-Document Retrieval & Qdrant Cloud Native Hybrid Architecture
* **Goal:** Eliminate vector dilution and clause orphaning in legal insurance policies by decoupling search resolution (`ChildChunk` ~80–150 words) from LLM synthesis context (`ParentSection` ~600–1,500 words), migrating vector and lexical storage into Qdrant Cloud with server-side RRF fusion.
* **Status:** **COMPLETE (100%) ✅**
  - **Universal Layout Parser**: `src/parser.py` parses standardized Markdown into `ParentSection` and `ChildChunk` objects using heading typography (`#`, `##`, `###`), preserving Markdown tables intact without cell severance.
  - **Cloud-Native BM25 Sparse Vectorizer**: `src/bm25_vectorizer.py` implements regex tokenization, question stopword pruning, term frequency weighting, and sparse vector generation for Qdrant.
  - **Qdrant Cloud Storage Manager**: `src/storage.py` manages cloud collection `insurance_chunks` with named vectors (`dense` 768-d via `BAAI/bge-base-en-v1.5` on NVIDIA GPU + `bm25` sparse), payload indexes on `card_id`, and denormalized `parent_content`.
  - **Cloud Hybrid Retriever**: `src/retriever.py` executes single-call `query_points` in Qdrant Cloud using server-side RRF fusion ($k=60$), followed by parent deduplication returning top 2–3 unique parent sections.
  - **Synthesis Integration**: `src/generator.py` and `api/main.py` updated to stream parent-level context to the LLM and render exact page citations in the web UI.
  - **Master Cloud Indexer**: `scripts/index_all_cards.py` indexes all card booklets into Qdrant Cloud.
  - **Benchmark Precision**: Automated benchmarks for CIBC (`tests/test_benchmark_cibc.py`) and TD (`tests/test_benchmark_td.py`) passing with 100% precision.
  - **Full Automated Test Suite**: 37/37 tests passing in `pytest tests/ -v`.
