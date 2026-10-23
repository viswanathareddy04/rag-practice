# Credit Card Insurance RAG: System Phase Architecture

This directory contains standalone, in-depth architectural guides for each development phase of the Bank-Agnostic Credit Card Insurance RAG System.

---

## 1. System Overview & End-to-End Pipeline

```mermaid
flowchart LR
    subgraph Phase 1: Ingestion
        RawPDF[Raw PDFs in data/] --> Registry[Card Registry]
        Registry --> Parser[InsuranceDocParser]
        Parser --> Chunks[Contextual Breadcrumb Chunks]
    end

    subgraph Phase 2: Hybrid Retrieval
        Chunks --> Storage[StorageManager]
        Storage --> BM25[(BM25Plus Index)]
        Storage --> Dense[(Dense MiniLM Vectors)]
        Query[User Question] --> Retriever[HybridRetriever]
        BM25 --> Retriever
        Dense --> Retriever
        Retriever --> RRF[RRF Fusion Top-K Chunks]
    end

    subgraph Phase 3: Grounded Synthesis
        RRF --> Generator[LLMGenerator]
        Generator --> Prompt[Strict Grounding Prompt]
        Prompt --> Ollama[Local Ollama / Cloud LLM]
        Ollama --> Citations[Structured Guardrailed Response]
    end

    subgraph Phase 4: Full-Stack Web App
        Citations --> FastAPI[FastAPI Backend]
        FastAPI --> WebUI[Vanilla HTML/CSS/JS UI]
    end
```

---

## 2. Directory & Phase Map

| Phase | Title | Focus Area | Deliverables | Status | Documentation |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Phase 1** | Ingestion & Breadcrumb Parsing | Document parsing, outline mapping, hierarchical tagging | `src/registry.py`, `src/parser.py`, `scripts/inspect_policies.py` | **DONE (100%)** ✅ | [Phase 1 Guide](PHASE_1_INGESTION.md) |
| **Phase 2** | Hybrid Retrieval & Benchmark | Dual-index search (BM25Plus + MiniLM), RRF, ground truth | `src/storage.py`, `src/retriever.py`, `tests/test_benchmark_td.py` | **READY TO RUN** 🚀 | [Phase 2 Guide](PHASE_2_RETRIEVAL.md) |
| **Phase 3** | Grounded Synthesis & Guardrails | Local LLM inference, zero-hallucination prompt, citations | `src/generator.py`, prompt template, Ollama runner | **NOT STARTED** ⏹️ | [Phase 3 Guide](PHASE_3_SYNTHESIS.md) |
| **Phase 4** | Full-Stack Web App & Multi-Bank | FastAPI service, responsive UI, multi-bank validation | `api/main.py`, `frontend/`, CIBC/Scotia validation | **NOT STARTED** ⏹️ | [Phase 4 Guide](PHASE_4_FULLSTACK_UI.md) |

---

## 3. Core Architectural Tenets Across All Phases

1. **Bank Agnosticism**: No deterministic regex rules specific to any bank (e.g. no `if "td" in card_id:`). The system relies on layout patterns, structural bookmarks, and generalized text normalizations.
2. **Context Preservation**: Exclusions and coverage caps are meaningless in isolation. Every piece of text is bound to its card, policy, section, and page number via breadcrumbs.
3. **Evidence-Based Grounding**: Every answer generated must cite exact certificate pages and sections. Unverified assumptions are strictly rejected.
4. **Local-First & Production-Ready**: Designed to operate completely offline using local CPU/GPU resources via Ollama and `sentence-transformers`, with seamless toggles for cloud APIs.
