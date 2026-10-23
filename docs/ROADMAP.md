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
| **Phase 2** | Hybrid Retrieval & Benchmark | Card-scoped BM25Plus + Dense RRF store, 10/10 tests pass | **READY TO RUN** 🚀 | [Phase 2 Guide](phases/PHASE_2_RETRIEVAL.md) |
| **Phase 3** | Grounded Synthesis & Guardrails | Local Ollama LLM integration, zero-hallucination prompt | **NOT STARTED** ⏹️ | [Phase 3 Guide](phases/PHASE_3_SYNTHESIS.md) |
| **Phase 4** | Web Interface & Multi-Bank | FastAPI service, Vanilla HTML/CSS/JS UI, Scotia/CIBC validation | **NOT STARTED** ⏹️ | [Phase 4 Guide](phases/PHASE_4_FULLSTACK_UI.md) |

> **Architectural Changes & History**: See [Design Change Records](design_changes/README.md) (e.g. [DCR-001: Phase 1 Parser Refactor](design_changes/001_phase1_parser_hierarchical_tracking.md)).

---

## 2. Detailed Status by Phase

### Phase 1: Ingestion & Hierarchical Breadcrumb Parsing
* **Goal:** Ingest any bank's digital insurance PDF, preserve markdown tables and headings, and prepend hierarchical breadcrumbs (`[Card] > [Policy] > [Section] (Page X)`) without bank-specific deterministic regex.
* **What is complete:**
  - `src/registry.py`: `CardRegistry` dynamically discovers bank folders (`cibc/`, `scotia/`, `td/`) and formats human-readable card names and normalized IDs. Unit tested & passing (`test_registry.py`).
  - `src/parser.py`: Layout-aware extraction via `pymupdf4llm` with PDF outline (`doc.get_toc()`) integration. Automatically maps page spans and intra-page section transitions (`#`, `##`, `###`, `Section X - ...`).
  - Unicode Normalization: Normalizes special non-breaking hyphens (`\u2011`), curly quotes, and non-breaking spaces so text matching is reliable.
  - Page Citation Accuracy: Cleanly maps true document pages (1-52) and prevents previous policies from bleeding into transition pages.
  - Policy Hierarchy Inspection Tool: `scripts/inspect_policies.py` verifies all 10 Level-1 policies, distinct sections, and formatted breadcrumbs from `data/td/TD Aeroplan Visa Infinite Privilege Card COI.pdf`.
  - Comprehensive Unit Tests: `tests/test_parser.py` passing (3/3 tests covering multi-section pages, policy boundaries, and breadcrumbs).
* **Sign-Off Status:** **COMPLETE ✅**

---

### Phase 2: Hybrid Retrieval & Ground Truth Benchmark
* **Goal:** Deliver reliable, millisecond search combining lexical BM25 (for exact numbers, limits, and policy terms) and dense semantic vectors (`sentence-transformers/all-MiniLM-L6-v2`) via Reciprocal Rank Fusion (RRF).
* **What is complete:**
  - `src/storage.py`: `StorageManager` persists chunks (`chunks.json`), BM25Plus index (`bm25.pkl`), and normalized embeddings (`embeddings.npy`) per card. Unit tested & passing.
  - `src/retriever.py`: `HybridRetriever` implements dual-index retrieval with RRF ranking. Unit tested & passing.
* **Remaining Work for Phase 2 Sign-Off:**
  - Re-run benchmark with the finalized Phase 1 parser to verify 10/10 automated tests pass on TD Aeroplan VIP.

---

### Phase 3: Grounded Synthesis with Citation Guardrails
* **Goal:** Connect local Ollama (`qwen2.5:7b` / `llama3.2`) with a strict grounding prompt that outputs:
  1. Direct Answer
  2. Coverage Limits & Details (dollar amounts, duration caps)
  3. Key Conditions & Exclusions (pre-existing conditions, age brackets)
  4. Certificate Citations (`[Policy, Section, Page X]`)
* **Status:** Generator logic drafted in `src/generator.py` with mock testing passing. Ready for live Ollama integration once Phase 2 is verified.

---

### Phase 4: Full-Stack Web UI & Cross-Bank Verification
* **Goal:** FastAPI backend + modern Web UI (card dropdown, streaming chat, collapsible source inspector) and verifying end-to-end functionality across CIBC and Scotiabank cards.
* **Status:** Planned; implementation begins once Phase 3 terminal Q&A is verified.
