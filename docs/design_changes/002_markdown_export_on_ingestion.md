# Design Change Record: Markdown Document Export on Ingestion

**ID:** DCR-002  
**Date:** 2026-10-23  
**Status:** Implemented & Verified ✅  
**Component:** `src/parser.py`, `tests/test_parser.py`, `data/extracted/`  
**Author:** AI Agent & Engineering Partner  

---

## 1. Context & Motivation

Previously, `InsuranceDocParser` extracted Markdown representations of PDF pages into memory, broke them into `DocumentChunk` dataclasses, and discarded the full continuous document representation. While chunks were persisted to `.index/<card_id>/chunks.json`, developers, QA engineers, and domain specialists had no direct way to inspect or audit the complete converted Markdown representation of the certificate.

To increase inspection transparency and enable human review of extracted tables and Markdown layout fidelity, we added automatic Markdown persistence during ingestion.

---

## 2. Design Change

1. **`parse_pdf` Signature Update**:
   ```python
   def parse_pdf(
       self, 
       pdf_path: str, 
       card_name: str, 
       card_id: Optional[str] = None,
       extract_dir: Optional[str] = "data/extracted"
   ) -> List[DocumentChunk]:
   ```
2. **Dedicated Exporter (`_save_extracted_markdown`)**:
   - Compiles all page Markdown streams into a single document.
   - Prepends metadata header (Card name, card ID, total page count).
   - Annotates page transitions with HTML comments (`<!-- Page X -->`).
   - Writes the file to `data/extracted/<card_id>.md` using UTF-8 encoding.

---

## 3. Verification & Evidence

- **Unit Test**: `tests/test_parser.py::test_save_extracted_markdown_to_disk` (Passed).
- **Artifact Verified**: `data/extracted/td_aeroplan_visa_infinite_privilege_card_coi.md` (221,515 bytes, 3,426 lines across 52 pages).
