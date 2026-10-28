# Credit Card Insurance RAG: Phased Roadmap & Live Status

**Last Updated:** 2026-10-28  
**Architecture:** Bank-Agnostic Two-Tier Parent-Document Retrieval with Qdrant Cloud Native Hybrid Search  
**Active Test Suite:** 37/37 Automated Tests Passing (100% Pass Rate)  
**Target Baseline Cards:** TD Aeroplan VIP (52 pages), CIBC Aeroplan VIP, Scotiabank Passport VIP  
**Input Data Directory:** `data/` (Strictly dedicated to raw input PDFs: `cibc/`, `scotia/`, `td/`)  
**Standardized Markdown:** `data/extracted/` (Standardized continuous Markdown per card)  
**Project Root:** `d:\rag-practice\` (`src/`, `tests/`, `docs/`, `api/`, `frontend/`, `scripts/`)  

---

## 1. Phase Status Overview

| Phase | Title & Focus Area | Deliverables & Core Technologies | Status | Verification & Benchmarks | Detailed Guide |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Phase 1** | **Ingestion & Universal Layout Parsing** | `src/registry.py`, `src/parser.py`, `data/extracted/*.md`, `scripts/inspect_policies.py`<br>*(PyMuPDF, pymupdf4llm, Unicode normalizer)* | **COMPLETE (100%)** ✅ | 6/6 tests passing (`tests/test_parser.py`, `tests/test_registry.py`). All 10 TD baseline policies verified without bank regex. | [Phase 1 Guide](phases/PHASE_1_INGESTION.md) |
| **Phase 2** | **Hybrid Retrieval & Benchmark Baseline** | `src/storage.py` (Local fallback), `src/retriever.py`, local RRF fusion ($k=60$)<br>*(BM25Plus, sentence-transformers/all-MiniLM-L6-v2)* | **COMPLETE (100%)** ✅ | 10/10 TD benchmark tests passing (`tests/test_benchmark_td.py`). Stopword pruning and exact number matching. | [Phase 2 Guide](phases/PHASE_2_RETRIEVAL.md) |
| **Phase 3** | **Grounded Synthesis & Citation Guardrails** | `src/generator.py`, `scripts/query.py`, `scripts/eval_synthesis.py`<br>*(qwen2.5:7b via local Ollama, 4-part response schema)* | **COMPLETE (100%)** ✅ | 4/4 generator tests passing (`tests/test_generator.py`). 5/5 synthesis evaluation cases pass with 100% negative rejection. | [Phase 3 Guide](phases/PHASE_3_SYNTHESIS.md) |
| **Phase 4** | **Full-Stack Web App & Multi-Bank Verification** | `api/main.py`, `frontend/` (Dark Fintech UI), `tests/test_multibank.py`<br>*(FastAPI, Uvicorn, SSE streaming, Vanilla HTML5/CSS3/ES6)* | **COMPLETE (100%)** ✅ | 4/4 API tests + 3/3 multi-bank tests passing (`tests/test_api.py`, `tests/test_multibank.py`). Zero build step. | [Phase 4 Guide](phases/PHASE_4_FULLSTACK_UI.md) |
| **Phase 5** | **Parent-Document Retrieval & Qdrant Cloud Hybrid** | `src/bm25_vectorizer.py`, `src/storage.py` (`QdrantStorageManager`), `src/retriever.py` (`HybridParentRetriever`), `scripts/index_all_cards.py`<br>*(Qdrant Cloud, BAAI/bge-base-en-v1.5 768-d on GPU, server-side RRF)* | **COMPLETE (100%)** ✅ | **37/37 tests passing** in full test suite. CIBC (`$100,000` MSRP, medical duration) & TD benchmarks verified with 100% precision. | [Phase 5 Guide](phases/PHASE_5_PARENT_DOCUMENT_QDRANT.md) |
| **Phase 6** | **Multi-Card Policy Comparison Engine** | Multi-card comparative synthesis prompt, side-by-side policy matrix UI, `/api/compare` endpoint | **PLANNED** 📋 | Comparative benchmark across TD vs CIBC vs Scotiabank coverage limits. | — |
| **Phase 7** | **Autonomous Claim Eligibility Auditor** | Incident intake questionnaire, multi-clause exclusion auditor, payout estimation engine | **BACKLOG** ⏳ | Ground-truth claim scenario evaluations (medical, baggage, trip cancellation). | — |

> **Architectural Changes & History**: See [Design Change Records](design_changes/README.md) (e.g. [DCR-001: Phase 1 Parser Refactor](design_changes/001_phase1_parser_hierarchical_tracking.md), [DCR-002: Markdown Document Export](design_changes/002_markdown_export_on_ingestion.md), [DCR-003: Parent-Document Qdrant Cloud Migration](design_changes/003_parent_document_qdrant_cloud.md)).

---

## 2. Detailed Status by Phase

### Phase 1: Ingestion & Universal Document Layout Parsing
* **Goal:** Ingest heterogeneous credit card insurance booklets (20–60 page PDFs) from any bank, extract clean Markdown with tables and headings intact, track line-by-line physical page numbers, and prepend hierarchical breadcrumbs (`[Card] > [Policy] > [Section] (Page X)`) without bank-specific deterministic regex.
* **Status:** **COMPLETE (100%) ✅**
  - **Dynamic Card Discovery**: `src/registry.py` scans `data/` subdirectories (`cibc/`, `scotia/`, `td/`) and produces normalized identifiers and metadata.
  - **Universal Layout Parser**: `src/parser.py` parses layout typography using `pymupdf4llm` with page comments (`<!-- Page X -->`).
  - **Continuous Markdown Export**: Automatically exports full converted documents to `data/extracted/<card_id>.md` for inspection.
  - **Boundary Isolation**: Deferral logic guarantees policy and section headers transition on the exact line of occurrence without leaking across page boundaries.

---

### Phase 2: Hybrid Retrieval & Ground Truth Benchmark Baseline
* **Goal:** Deliver reliable, millisecond search combining lexical BM25 (for exact numbers, limits, and policy terms) and dense semantic vectors via Reciprocal Rank Fusion (RRF).
* **Status:** **COMPLETE (100%) ✅**
  - **Offline Fallback Storage**: `StorageManager` in `src/storage.py` persists `.index/<card_id>/` disk artifacts (`chunks.json`, `bm25.pkl`, `embeddings.npy`).
  - **Dual-Index RRF Retriever**: `HybridRetriever` in `src/retriever.py` executes dual lexical and semantic searches with question stopword suppression.
  - **Benchmark Precision**: 10/10 benchmark test cases passing with 100% precision (`tests/test_benchmark_td.py`).
  - *Phase 5 Evolution*: Elevated in Phase 5 to `QdrantStorageManager` + `HybridParentRetriever` with cloud-native BM25 and server-side RRF.

---

### Phase 3: Grounded Synthesis with Citation Guardrails
* **Goal:** Connect local Ollama (`qwen2.5:7b`) with a strict grounding prompt and 4-part structured response schema with exact page citations and zero hallucination tolerance.
* **Status:** **COMPLETE (100%) ✅**
  - **Grounded Generator**: `src/generator.py` streams tokens from local Ollama (`qwen2.5:7b`) with graceful offline fallback.
  - **Terminal Client**: Standalone interactive terminal client `scripts/query.py` with live token streaming and ANSI formatting.
  - **Automated Synthesis Evaluation**: `scripts/eval_synthesis.py` asserts 100% schema compliance and negative rejection (e.g. explicitly rejecting pet emergencies).
  - *Phase 5 Evolution*: In Phase 5, synthesis context was upgraded from flat chunks to complete, unsevered `ParentSection`s.

---

### Phase 4: Full-Stack Web App & Cross-Bank Verification
* **Goal:** FastAPI backend + modern Dark Fintech Web UI (card dropdown, streaming chat, collapsible source inspector) and verifying end-to-end functionality across TD, CIBC, and Scotiabank certificates.
* **Status:** **COMPLETE (100%) ✅**
  - **FastAPI Microservice**: `api/main.py` serves `/api/status`, `/api/cards`, `/api/query` (SSE streaming tokens and structured citations), `/api/index/{card_id}`, and mounts `frontend/` on `http://localhost:8001/`.
  - **Dark Fintech Web UI**: `frontend/` in pure Vanilla HTML5, CSS custom properties, and ES6 JS (`#080C14` background, glassmorphism, real-time token streaming, citation drawer). Zero build step (`npm run build` not required).
  - **Cross-Bank Generalization**: `tests/test_multibank.py` verifies discovery, ingestion, and hybrid retrieval across CIBC and Scotiabank certificates.

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
  - **Full Automated Test Suite**: **37/37 tests passing** in `pytest tests/ -v`.

---

## 3. Upcoming Phases & Architectural Roadmap

### Phase 6: Multi-Card Policy Comparison Engine (Planned 📋)
* **Objective:** Enable users to select 2 or 3 credit cards simultaneously (e.g. *TD Aeroplan VIP* vs *CIBC Aeroplan VIP*) and compare specific coverage riders (e.g. rental car collision limits, emergency travel medical duration, mobile device deductibles) side by side.
* **Deliverables:**
  - Multi-card parallel hybrid retrieval pipeline in `src/retriever.py`.
  - Structured comparative prompt template in `src/generator.py` outputting a Markdown comparison matrix.
  - New API endpoint `POST /api/compare` accepting `card_ids: List[str]` and `query: str`.
  - Frontend comparison modal rendering synchronized side-by-side table columns.

---

### Phase 7: Autonomous Claim Eligibility Auditor (Backlog ⏳)
* **Objective:** An interactive audit assistant where cardholders input an incident scenario (e.g. *"Flight was delayed 5 hours due to mechanical issues, baggage delayed 14 hours, paid with card"*), and the system audits the exact Certificate of Insurance to output:
  1. Claim Eligibility Verdict (Approved / Denied / Additional Documentation Needed).
  2. Maximum Payable Reimbursement (itemized by benefit).
  3. Strict Filing Deadlines & Mandatory Required Receipts.
