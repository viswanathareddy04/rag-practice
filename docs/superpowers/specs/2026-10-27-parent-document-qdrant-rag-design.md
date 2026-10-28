# Architecture Design Spec: Parent-Document Retrieval with Qdrant Cloud Hybrid Search

**Date:** 2026-10-27  
**Status:** Approved by User / Ready for Implementation Planning  
**Target Card Benchmark:** CIBC Aeroplan Visa Infinite Privilege Card & TD Aeroplan Visa Infinite Privilege Card  
**Expansion Scope:** Universal & Bank-Agnostic across 20–30+ Credit Card Insurance Booklets  
**Hardware Profile:** Local NVIDIA GeForce RTX 5070 (12 GB VRAM) + Qdrant Cloud Cluster (`ca-central-1-0.aws.cloud.qdrant.io`)

---

## 1. Executive Summary & Problem Definition

### 1.1 The Problem
In legal credit card insurance contracts, flat chunking creates two catastrophic failure modes:
1. **Clause Orphaning & Context Fragmentation**: Isolated sentences (e.g. *"Coverage is capped at $1,000 for essential purchases after a 4-hour delay"*) lack the parent policy context, causing semantic vectors to miss the clause or conflate *Flight Delay* with *Baggage Delay*.
2. **Severed Conditions & Lost Caveats**: Critical conditions (e.g., pre-existing condition stability periods, exclusions, deductible tiers) often span multiple sentences or paragraphs. If a retriever retrieves only 1 chunk from an exclusion section, the generative LLM answers without knowing about the mandatory exceptions.

### 1.2 The Solution: Parent-Document Retrieval (Small-to-Big) with Qdrant Cloud
We evolve the ingestion and retrieval architecture into a **Two-Tier Parent-Document System** backed by a single **Qdrant Cloud** cluster with native **Dense + Sparse (BM25) Hybrid Search** and server-side **Reciprocal Rank Fusion (RRF)**:
- **Child Chunks (~100–150 words)**: Serve as granular "Search Beacons", indexed in Qdrant with dense vectors (`BAAI/bge-base-en-v1.5`, 768-d) and BM25 sparse vectors.
- **Parent Sections (~600–1,500 words)**: Serve as complete "Synthesis Context", stored in the Qdrant point payloads.
- **Single-Hop Cloud Resolution**: Top child search hits already contain their parent section in their payload—eliminating secondary network round-trips.
- **Deduplication & Assembly**: Top child hits are deduplicated by `parent_id` into the top 2–3 unique Parent Sections and fed to local `qwen2.5:7b` via Ollama for zero-hallucination synthesis.
- **Child-Highlighted Citations**: The UI citation drawer highlights the exact matching child clause and page number, while the LLM answers from the full parent section.

---

## 2. Core Architectural Principles & Invariants

1. **Strictly Bank-Agnostic (Zero Hardcoded Bank Rules)**:
   - The parser must NEVER use bank-specific regex or `if bank == 'cibc'` branching.
   - All hierarchy is discovered purely through universal Markdown structures (`#`, `##`, `###`, tables `|`, and double newlines `\n\n`) output by `pymupdf4llm`.
2. **100% Cloud-Native Search & Stateless Backend**:
   - Both dense vectors and BM25 sparse vectors, along with document payloads, reside in Qdrant Cloud.
   - No local `.index/` directory or `bm25.pkl` files required on disk.
   - The FastAPI backend only requires `QDRANT_URL` and `QDRANT_API_KEY`.
3. **GPU-Accelerated Embeddings**:
   - Takes full advantage of the local NVIDIA GeForce RTX 5070 (12 GB VRAM) using `BAAI/bge-base-en-v1.5` (768-d), with automatic fallback to CPU.
4. **Idempotent Ingestion**:
   - Re-indexing any card PDF deletes prior points matching `card_id` before inserting new points, preventing duplicate ghost chunks.
5. **Decoupled Search vs Synthesis**:
   - Search resolution: 100–150 words (high semantic density, no vector dilution).
   - Synthesis resolution: 600–1,500 words (complete section legal jurisdiction).

---

## 3. Data Models & Contracts

### 3.1 Data Structures in `src/parser.py`

```python
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class ParentSection:
    parent_id: str          # Format: f"{card_id}__{policy_slug}__{section_slug}"
    card_id: str            # e.g., "cibc_aeroplan_vip"
    card_name: str          # e.g., "CIBC Aeroplan Visa Infinite Privilege Card"
    policy_name: str        # e.g., "OUT-OF-PROVINCE EMERGENCY TRAVEL MEDICAL INSURANCE"
    section_title: str      # e.g., "SECTION 8 - WHAT ARE YOU NOT COVERED FOR?"
    page_start: int         # Starting PDF page (1-indexed)
    page_end: int           # Ending PDF page (1-indexed)
    content: str            # Full Markdown text of the section (~600–1,500 words)

@dataclass
class ChildChunk:
    child_id: str           # Format: f"{parent_id}__c{index:03d}"
    parent_id: str          # Foreign key referencing ParentSection.parent_id
    card_id: str
    card_name: str
    policy_name: str
    section_title: str
    page_number: int        # Exact PDF page where this child clause appears
    content: str            # Specific clause/paragraph text (~100–150 words)
    breadcrumb: str         # "[Card: Name] > [Policy] > [Section] (Page X)"
    parent_content: str     # Denormalized copy of the full parent section text
```

---

## 4. Ingestion Pipeline & Universal Structural Parser

```
Raw PDF (data/<bank>/*.pdf)
  │
  ▼ [fitz & pymupdf4llm]
Standardized Markdown with Tables & Page Tags (<!-- Page X -->)
  │
  ▼ [Universal Heading Stack Tracker]
Identify Level-1/2 Policies (#, ##) and Level-3 Sections (###)
  │
  ▼
ParentSection List (Boundaries across pages, uninterrupted text)
  │
  ▼ [Markdown Block Chunker]
  ├── Tables: Kept row-by-row or row-clusters (never split across '|')
  ├── Lists: Grouped by bullet clusters up to ~150 words
  └── Paragraphs: Split on '\n\n', fallback to sentence boundaries ('. ')
  │
  ▼
ChildChunk List (~100–150 words each, carrying parent_id & parent_content)
```

### 4.1 Heading Stack & Parent Section Formation
1. As the parser reads markdown lines:
   - Level 1 / Level 2 (`#`, `##`, or TOC bookmarks): Updates active `policy_name`.
   - Level 3 (`###` or `####` in policy body): Flushes previous section buffer and starts a new `ParentSection`.
2. Heading sanitization strips HTML tags (`<mark>`, `<u>`, `<b>`) and Markdown formatting (`*`, `_`, `#`) before generating clean slugified IDs.
3. If no Level 3 heading exists on early pages (e.g. Summary of Benefits tables on pages 2–3), the table block is assigned its own `ParentSection` (e.g., `Summary of Benefits`).

### 4.2 Structural Child Splitting
Within each `ParentSection`:
- **Table Blocks**: Identified by lines starting with `|`. Table headers are preserved, and rows are chunked without severing cells.
- **Bullet Lists & Paragraphs**: Split by `\n\n`. Blocks under 150 words form single child chunks. Blocks over 150 words split on sentence regex `(?<=[.!?])\s+`.
- Each child chunk records its exact 1-indexed `page_number` using page comment markers (`<!-- Page X -->`).

---

## 5. Storage Layer & Qdrant Cloud Schema

### 5.1 Collection Configuration in `src/storage.py`
A single collection in Qdrant Cloud handles all cards:
- **Collection Name**: `insurance_chunks`
- **Dense Vector Config**:
  - `name`: `"dense"`
  - `size`: 768 (`BAAI/bge-base-en-v1.5`)
  - `distance`: `Distance.COSINE`
- **Sparse Vector Config**:
  - `name`: `"bm25"`
  - `index`: `models.SparseIndexParams()`
- **Payload Schema**:
  ```json
  {
    "card_id": "cibc_aeroplan_vip",
    "card_name": "CIBC Aeroplan Visa Infinite Privilege Card",
    "bank": "cibc",
    "child_id": "cibc_aeroplan_vip__travel_med__sec8__c002",
    "child_content": "An exclusion applies for any medical condition that was not stable before your trip...",
    "page_number": 5,
    "breadcrumb": "[Card: CIBC Aeroplan VIP] > [Travel Medical] > [SECTION 8 - WHAT ARE YOU NOT COVERED FOR?] (Page 5)",
    "parent_id": "cibc_aeroplan_vip__travel_med__sec8",
    "parent_policy": "OUT-OF-PROVINCE EMERGENCY TRAVEL MEDICAL INSURANCE",
    "parent_section": "SECTION 8 - WHAT ARE YOU NOT COVERED FOR?",
    "parent_page_span": "4-9",
    "parent_content": "### SECTION 8 - WHAT ARE YOU NOT COVERED FOR?\nFull text of the entire 1,100-word section..."
  }
  ```

### 5.2 Scalability & Payload Indexes
- Create a `KEYWORD` payload index on `card_id` in Qdrant Cloud.
- Enables single-card pre-filtering:
  ```python
  query_filter = models.Filter(
      must=[models.FieldCondition(key="card_id", match=models.MatchValue(value=card_id))]
  )
  ```
- Ensures sub-millisecond search across 20–30 cards (7,500+ vectors).

### 5.3 BM25 Sparse Vectorizer in Python
- Computes BM25 term frequency weights for each token in the child text:
  $$W(t, D) = \text{IDF}(t) \cdot \frac{f(t, D) \cdot (k_1 + 1)}{f(t, D) + k_1 \cdot (1 - b + b \cdot \frac{|D|}{\text{avgdl}})}$$
- Tokens are hashed to 32-bit positive integers (`zlib.crc32(token.encode()) % (2**31 - 1)`).
- Uploaded as `models.SparseVector(indices=[...], values=[...])`.

---

## 6. Retrieval Engine & Prompt Assembly

### 6.1 Hybrid Cloud Retrieval in `src/retriever.py`
When querying:
1. **Encode Query**:
   - `query_dense`: `embedder.encode([query], device=device)[0]` (768-dim float array).
   - `query_sparse`: BM25 term tokenizer with question-stopword pruning (`what`, `is`, `the`, `can`, `i`, etc.).
2. **Execute Single Cloud Query**:
   ```python
   response = qdrant_client.query_points(
       collection_name="insurance_chunks",
       prefetch=[
           models.Prefetch(query=query_dense, using="dense", filter=card_filter, limit=20),
           models.Prefetch(query=query_sparse, using="bm25", filter=card_filter, limit=20),
       ],
       query=models.FusionQuery(fusion=models.Fusion.RRF),
       limit=8
   )
   ```
3. **Parent Deduplication**:
   - Extract unique `parent_id`s in rank order from the top child hits.
   - Select top 2–3 unique Parent Sections (totaling ~1,500–2,500 words).
4. **Citation Extraction**:
   - The top matching child chunks provide exact page citations and evidence snippets for the web UI drawer.

### 6.2 Prompt Assembly for Ollama (`qwen2.5:7b`) in `src/generator.py`
Context is constructed using unbroken parent sections:
```text
CONTEXT INFORMATION:

--- Parent Section 1: [OUT-OF-PROVINCE EMERGENCY TRAVEL MEDICAL INSURANCE] > SECTION 8 - WHAT ARE YOU NOT COVERED FOR? (Pages 7-9) ---
[Full, uninterrupted text of Section 8 including pre-existing conditions, stability clauses, sports exclusions]

--- Parent Section 2: [OUT-OF-PROVINCE EMERGENCY TRAVEL MEDICAL INSURANCE] > SECTION 5 - WHEN DOES COVERAGE BEGIN AND END? (Pages 5-6) ---
[Full, uninterrupted text of Section 5 including 31-day vs 10-day duration rules by age]

USER QUESTION:
Are my pre-existing medical conditions covered if I travel for 2 weeks?
```

The LLM produces the 4-part schema:
- `### Direct Answer`
- `### Coverage Limits & Details`
- `### Key Conditions & Exclusions`
- `### Policy Citations`

---

## 7. Configuration, Testing & Verification Plan

### 7.1 Environment Variables (`.env`)
```ini
QDRANT_URL=https://bebb4a26-dfc6-47bb-958a-c75d097674e5.ca-central-1-0.aws.cloud.qdrant.io
QDRANT_API_KEY=your_key_here
EMBEDDING_MODEL=BAAI/bge-base-en-v1.5
EMBEDDING_DIM=768
EMBEDDING_DEVICE=cuda
LLM_MODEL=qwen2.5:7b
OLLAMA_BASE_URL=http://localhost:11434
```

### 7.2 Automated Test Suite
1. **`tests/test_parser.py`**:
   - Verify `ParentSection` generation from Markdown headings.
   - Verify `ChildChunk` sizes (100–150 words) and `parent_id` linking.
   - Verify zero table row splitting on both TD and CIBC PDFs.
2. **`tests/test_storage.py`**:
   - Verify Qdrant Cloud client initialization (with in-memory fallback for offline test execution).
   - Verify named dense (768-d) and sparse BM25 vectors.
   - Verify payload indexing on `card_id`.
   - Verify idempotent upserts (delete before insert).
3. **`tests/test_retriever.py`**:
   - Verify query encoding (dense + sparse).
   - Verify Qdrant Cloud RRF fusion.
   - Verify child-to-parent deduplication logic.
4. **`tests/test_benchmark.py`**:
   - Assert key insurance thresholds:
     - CIBC Car Rental MSRP: `$100,000` (48 days).
     - TD Car Rental MSRP: `$85,000` (48 days).
     - CIBC Medical coverage duration: `31 days` (age <= 64) vs `10 days` (age >= 65).
     - TD Mobile Device deductible: tiered schedule up to `$1,500`.
     - Negative controls: Explicitly rejects pet medical emergencies.
