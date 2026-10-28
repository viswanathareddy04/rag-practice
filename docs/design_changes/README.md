# Design Change Records (DCR)

This directory maintains formal Design Change Records documenting architectural decisions, design modifications, component refactors, and problem diagnoses throughout the project lifecycle.

| ID | Title | Date | Component | Status |
| :--- | :--- | :--- | :--- | :--- |
| [DCR-001](001_phase1_parser_hierarchical_tracking.md) | Phase 1 Parser: Dual Outline & Layout Hierarchical Tracking | 2026-10-23 | `src/parser.py`, `tests/test_parser.py` | **Implemented & Verified** ✅ |
| [DCR-002](002_markdown_export_on_ingestion.md) | Automatic Markdown Document Export to `data/extracted/<card_id>.md` | 2026-10-23 | `src/parser.py`, `tests/test_parser.py` | **Implemented & Verified** ✅ |
| [DCR-003](003_parent_document_qdrant_cloud.md) | Parent-Document Retrieval & Qdrant Cloud Hybrid Migration | 2026-10-27 | `src/parser.py`, `src/storage.py`, `src/retriever.py`, `src/bm25_vectorizer.py` | **Implemented & Verified** ✅ |
