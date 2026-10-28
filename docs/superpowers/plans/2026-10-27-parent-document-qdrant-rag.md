# Parent-Document Retrieval with Qdrant Cloud Hybrid Search Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Transform the credit card insurance RAG pipeline into a bank-agnostic, two-tier Parent-Document Retrieval architecture backed by Qdrant Cloud with native Dense (BGE-base 768-d) + Sparse (BM25) hybrid search and server-side RRF fusion.

**Architecture:** 
1. `pymupdf4llm` extracts standardized Markdown. A universal heading-stack parser partitions text into `ParentSection`s (~600–1,500 words) and `ChildChunk`s (~100–150 words).
2. Qdrant Cloud stores child points with named vectors (`dense` 768-d + `bm25` sparse) and denormalized `parent_content` in the payload.
3. Queries execute hybrid search in a single cloud request (`query_points` with server-side RRF and `card_id` filter). Top child hits are deduplicated into top 2–3 unique parent sections for LLM synthesis (`qwen2.5:7b`), while child clauses serve as exact page-level citations.

**Tech Stack:** 
- Python 3.12+, `qdrant-client` 1.18+, `sentence-transformers` (`BAAI/bge-base-en-v1.5`), `pymupdf` / `pymupdf4llm`
- Local NVIDIA GeForce RTX 5070 (12 GB VRAM) for embeddings / Ollama LLM
- Qdrant Cloud Cluster (`ca-central-1-0.aws.cloud.qdrant.io`)
- FastAPI, Uvicorn, Vanilla JS/CSS

**Spec:** [`docs/superpowers/specs/2026-10-27-parent-document-qdrant-rag-design.md`](file:///d:/rag-practice/docs/superpowers/specs/2026-10-27-parent-document-qdrant-rag-design.md)

## Global Constraints

- Zero bank-specific deterministic branching (`if bank == 'td'`) or bank-specific regex.
- All documents must be parsed using universal Markdown typography (`#`, `##`, `###`, tables `|`, paragraphs `\n\n`).
- All vectors and lexical BM25 weights must be cloud-native in Qdrant Cloud with zero local `.index` dependencies in production.
- Offline and local testing must gracefully fall back to in-memory Qdrant (`:memory:`) or local embedded storage when `QDRANT_API_KEY` is not present.
- Every child chunk must be between 80–150 words and never severed across Markdown table rows.
- Every task must include automated unit/integration tests following strict TDD.

---

### Task 1: Universal Layout Parser & Data Structures (`ParentSection` & `ChildChunk`)

**Files:**
- Modify: `src/parser.py`
- Modify: `tests/test_parser.py`

**Interfaces:**
- Consumes: Raw PDF path, `card_name`, `card_id`
- Produces: 
  - `ParentSection(parent_id, card_id, card_name, policy_name, section_title, page_start, page_end, content)`
  - `ChildChunk(child_id, parent_id, card_id, card_name, policy_name, section_title, page_number, content, breadcrumb, parent_content)`
  - Function: `InsuranceDocParser.parse_pdf(pdf_path, card_name, card_id) -> Tuple[List[ParentSection], List[ChildChunk]]`

- [ ] **Step 1: Write the failing tests for Universal Layout Parser**

```python
# tests/test_parser.py
from src.parser import InsuranceDocParser, ParentSection, ChildChunk

def test_parser_creates_parent_sections_and_children():
    parser = InsuranceDocParser()
    sample_md = """<!-- Page 1 -->
# TRAVEL MEDICAL INSURANCE
## CERTIFICATE OF INSURANCE
Introduction to the medical policy.

### SECTION 1 - SUMMARY OF BENEFITS
| Coverage | Maximum |
| Travel Medical | $5,000,000 |
| Trip Cancellation | $2,500 |

### SECTION 2 - LIMITATIONS AND EXCLUSIONS
This policy excludes pre-existing medical conditions that were not stable in the 90 days prior to departure. Stability means no new symptoms, no hospitalizations, and no change in medication or dosage.
Exclusion clause 2: Travelling against medical advice is not covered under any circumstances.
"""
    parents, children = parser.parse_markdown_text(
        sample_md, card_name="CIBC Aeroplan VIP", card_id="cibc_aeroplan_vip"
    )

    assert len(parents) >= 2
    assert all(isinstance(p, ParentSection) for p in parents)
    assert all(isinstance(c, ChildChunk) for c in children)

    # Check parent structure
    sec2 = next(p for p in parents if "LIMITATIONS AND EXCLUSIONS" in p.section_title)
    assert sec2.policy_name == "TRAVEL MEDICAL INSURANCE"
    assert "90 days prior to departure" in sec2.content

    # Check child structure and linking
    c_sec2 = [c for c in children if c.parent_id == sec2.parent_id]
    assert len(c_sec2) >= 1
    assert c_sec2[0].parent_content == sec2.content
    assert "[Card: CIBC Aeroplan VIP]" in c_sec2[0].breadcrumb

def test_parser_preserves_table_rows_intact():
    parser = InsuranceDocParser()
    sample_md = """<!-- Page 2 -->
### SUMMARY OF BENEFITS
| BENEFIT | LIMIT |
| Flight Delay | $500 per person |
| Baggage Delay | $1,000 per person |
"""
    parents, children = parser.parse_markdown_text(
        sample_md, card_name="TD Aeroplan VIP", card_id="td_aeroplan_vip"
    )
    table_children = [c for c in children if "|" in c.content]
    assert len(table_children) >= 1
    # Check that rows are intact and not broken across cells
    for tc in table_children:
        assert tc.content.count("|") >= 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_parser.py -v`  
Expected: FAIL with `AttributeError` or missing `ParentSection` / `parse_markdown_text`.

- [ ] **Step 3: Implement Universal Layout Parser in `src/parser.py`**

Refactor `src/parser.py` to:
1. Define `ParentSection` and `ChildChunk` dataclasses.
2. Implement clean text normalization (HTML tag stripping `<mark>`, `<u>`, `<b>`, unicode hyphen normalization).
3. Implement `parse_markdown_text` which parses `#`, `##`, `###` headings into a stack:
   - Level 1 / Level 2 (`#`, `##`) updates `policy_name`.
   - Level 3 / Level 4 (`###`, `####`) or Markdown table blocks start a new `ParentSection`.
4. Implement chunking of `ParentSection` into `ChildChunk`s:
   - Markdown table rows (`|`) are kept intact.
   - Text blocks split on `\n\n`, with long paragraphs split on sentence boundaries (`re.split(r'(?<=[.!?])\s+')`).
   - Every child links to `parent_id` and copies `parent_content`.
5. Update `parse_pdf()` to use `pymupdf4llm.to_markdown(doc, page_chunks=True)` and call `parse_markdown_text()`.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_parser.py -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/parser.py tests/test_parser.py
git commit -m "feat(parser): implement universal parent-document layout parser"
```

---

### Task 2: BM25 Sparse Vectorizer & Tokenizer

**Files:**
- Create: `src/bm25_vectorizer.py`
- Create: `tests/test_bm25_vectorizer.py`

**Interfaces:**
- Consumes: Raw text strings (corpus or query)
- Produces: 
  - `BM25Vectorizer.fit(corpus_texts: List[str])`
  - `BM25Vectorizer.encode_document(text: str) -> SparseVector(indices, values)`
  - `BM25Vectorizer.encode_query(query: str) -> SparseVector(indices, values)`

- [ ] **Step 1: Write the failing test for `BM25Vectorizer`**

```python
# tests/test_bm25_vectorizer.py
from src.bm25_vectorizer import BM25Vectorizer
from qdrant_client import models

def test_bm25_vectorizer_encodes_sparse_vectors():
    corpus = [
        "Car rental collision damage coverage covers up to $100,000 MSRP for 48 days.",
        "Flight delay insurance covers essential purchases up to $500 after 4 hours delay.",
        "Emergency travel medical insurance provides up to $5,000,000 for hospital stays."
    ]
    vectorizer = BM25Vectorizer(k1=1.5, b=0.75)
    vectorizer.fit(corpus)

    doc_vector = vectorizer.encode_document(corpus[0])
    assert isinstance(doc_vector, models.SparseVector)
    assert len(doc_vector.indices) > 0
    assert len(doc_vector.values) == len(doc_vector.indices)
    assert all(isinstance(idx, int) and idx > 0 for idx in doc_vector.indices)
    assert all(val > 0.0 for val in doc_vector.values)

def test_bm25_vectorizer_encodes_query_with_stopword_pruning():
    corpus = ["Rental car collision coverage covers damage up to $100,000."]
    vectorizer = BM25Vectorizer()
    vectorizer.fit(corpus)

    query_vector = vectorizer.encode_query("What is the maximum coverage for rental car?")
    assert isinstance(query_vector, models.SparseVector)
    assert len(query_vector.indices) > 0
    # Stopwords like 'what', 'is', 'the', 'for' should be pruned from sparse query
    tokens_encoded = [vectorizer.hash_token(t) for t in ["rental", "car", "coverage", "maximum"]]
    assert any(idx in tokens_encoded for idx in query_vector.indices)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_bm25_vectorizer.py -v`  
Expected: FAIL with `ModuleNotFoundError: No module named 'src.bm25_vectorizer'`

- [ ] **Step 3: Implement `BM25Vectorizer` in `src/bm25_vectorizer.py`**

```python
import math
import re
import zlib
from collections import Counter
from typing import List, Dict, Set
from qdrant_client import models

class BM25Vectorizer:
    STOP_WORDS: Set[str] = {
        "what", "is", "the", "for", "a", "an", "and", "or", "in", "on", "to",
        "of", "with", "at", "by", "do", "does", "how", "much", "many", "can", "i", "my"
    }

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_count = 0
        self.avg_doc_len = 0.0
        self.idf: Dict[str, float] = {}

    def tokenize(self, text: str) -> List[str]:
        return re.findall(r"\w+", text.lower())

    def hash_token(self, token: str) -> int:
        return zlib.crc32(token.encode("utf-8")) % (2**31 - 1)

    def fit(self, corpus: List[str]):
        self.doc_count = len(corpus)
        doc_lengths = []
        df: Counter = Counter()

        for doc in corpus:
            tokens = self.tokenize(doc)
            doc_lengths.append(len(tokens))
            unique_tokens = set(tokens)
            df.update(unique_tokens)

        self.avg_doc_len = sum(doc_lengths) / max(1, self.doc_count)
        self.idf = {}
        for token, freq in df.items():
            self.idf[token] = math.log(1.0 + (self.doc_count - freq + 0.5) / (freq + 0.5))

    def encode_document(self, text: str) -> models.SparseVector:
        tokens = self.tokenize(text)
        doc_len = len(tokens)
        counts = Counter(tokens)

        indices = []
        values = []
        for token, count in counts.items():
            idf_val = self.idf.get(token, 1.0)
            numerator = count * (self.k1 + 1.0)
            denominator = count + self.k1 * (1.0 - self.b + self.b * (doc_len / max(1.0, self.avg_doc_len)))
            weight = idf_val * (numerator / max(1e-6, denominator))
            if weight > 0:
                indices.append(self.hash_token(token))
                values.append(float(weight))

        return models.SparseVector(indices=indices, values=values)

    def encode_query(self, query: str) -> models.SparseVector:
        raw_tokens = self.tokenize(query)
        filtered = [t for t in raw_tokens if t not in self.STOP_WORDS]
        tokens = filtered if filtered else raw_tokens
        counts = Counter(tokens)

        indices = []
        values = []
        for token, count in counts.items():
            idf_val = self.idf.get(token, 1.0)
            indices.append(self.hash_token(token))
            values.append(float(idf_val * count))

        return models.SparseVector(indices=indices, values=values)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_bm25_vectorizer.py -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/bm25_vectorizer.py tests/test_bm25_vectorizer.py
git commit -m "feat(retrieval): implement cloud-native BM25 sparse vectorizer"
```

---

### Task 3: Qdrant Cloud Storage Layer & Batch Upsert

**Files:**
- Modify: `src/storage.py`
- Create: `tests/test_storage_qdrant.py`

**Interfaces:**
- Consumes: `QDRANT_URL`, `QDRANT_API_KEY`, `EMBEDDING_MODEL` (768-d), `List[ParentSection]`, `List[ChildChunk]`
- Produces: 
  - `QdrantStorageManager.create_collection_if_not_exists()`
  - `QdrantStorageManager.save_card_chunks(card_id, parent_sections, child_chunks)`
  - `QdrantStorageManager.is_indexed(card_id) -> bool`
  - `QdrantStorageManager.client: QdrantClient`

- [ ] **Step 1: Write the failing test for `QdrantStorageManager`**

```python
# tests/test_storage_qdrant.py
from qdrant_client import QdrantClient
from src.parser import ParentSection, ChildChunk
from src.storage import QdrantStorageManager

def test_qdrant_storage_manager_upserts_and_indexes():
    # Use in-memory client for testing
    client = QdrantClient(location=":memory:")
    storage = QdrantStorageManager(client=client, model_name="BAAI/bge-base-en-v1.5", device="cpu")

    parent = ParentSection(
        parent_id="cibc_vip__med__sec8",
        card_id="cibc_aeroplan_vip",
        card_name="CIBC Aeroplan VIP",
        policy_name="TRAVEL MEDICAL INSURANCE",
        section_title="SECTION 8 - WHAT ARE YOU NOT COVERED FOR?",
        page_start=7,
        page_end=9,
        content="Full text of Section 8..."
    )
    child = ChildChunk(
        child_id="cibc_vip__med__sec8__c001",
        parent_id=parent.parent_id,
        card_id=parent.card_id,
        card_name=parent.card_name,
        policy_name=parent.policy_name,
        section_title=parent.section_title,
        page_number=8,
        content="Pre-existing medical conditions not stable within 90 days are excluded.",
        breadcrumb="[Card: CIBC Aeroplan VIP] > [Medical] (Page 8)",
        parent_content=parent.content
    )

    storage.save_card_chunks(card_id="cibc_aeroplan_vip", parent_sections=[parent], child_chunks=[child])

    assert storage.is_indexed("cibc_aeroplan_vip")
    # Verify point count
    info = client.get_collection(storage.COLLECTION_NAME)
    assert info.points_count == 1

    # Verify payload content
    points, _ = client.scroll(collection_name=storage.COLLECTION_NAME, limit=1)
    p = points[0]
    assert p.payload["card_id"] == "cibc_aeroplan_vip"
    assert p.payload["parent_content"] == parent.content
    assert p.payload["page_number"] == 8
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_storage_qdrant.py -v`  
Expected: FAIL with missing `QdrantStorageManager` methods or attributes.

- [ ] **Step 3: Implement `QdrantStorageManager` in `src/storage.py`**

Refactor `src/storage.py` to:
1. Accept optional `client: Optional[QdrantClient]`. If `client is None`, read `QDRANT_URL` and `QDRANT_API_KEY` from environment. If `QDRANT_API_KEY` is missing, fallback to in-memory/local Qdrant.
2. Initialize `SentenceTransformer(model_name, device=device)`.
3. Initialize `BM25Vectorizer`.
4. `create_collection_if_not_exists`:
   - Configures named vectors: `dense` (768, Cosine) and `bm25` (SparseVectorParams).
   - Creates `KEYWORD` index on `card_id`.
5. `save_card_chunks(card_id, parent_sections, child_chunks)`:
   - Deletes existing points with `card_id` filter (idempotency).
   - Fits BM25 vectorizer on child texts.
   - Encodes dense embeddings in batches.
   - Encodes BM25 sparse vectors.
   - Generates deterministic UUID for each point: `uuid.uuid5(uuid.NAMESPACE_DNS, c.child_id)`.
   - Upserts points in batches with full payload (`parent_content`, `page_number`, `breadcrumb`, etc.).

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_storage_qdrant.py -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/storage.py tests/test_storage_qdrant.py
git commit -m "feat(storage): implement cloud Qdrant dual-vector storage manager"
```

---

### Task 4: Hybrid Cloud Retriever with Server-Side RRF & Parent Deduplication

**Files:**
- Modify: `src/retriever.py`
- Modify: `tests/test_retriever.py`

**Interfaces:**
- Consumes: `card_id: str`, `query: str`, `top_k_parents: int = 3`
- Produces: 
  - `RetrievedContext(parents: List[ParentSection], citations: List[Dict[str, Any]])`
  - Method: `HybridParentRetriever.retrieve(card_id, query) -> RetrievedContext`

- [ ] **Step 1: Write the failing test for `HybridParentRetriever`**

```python
# tests/test_retriever.py
from qdrant_client import QdrantClient
from src.parser import ParentSection, ChildChunk
from src.storage import QdrantStorageManager
from src.retriever import HybridParentRetriever, RetrievedContext

def test_retriever_hybrid_rrf_and_parent_deduplication():
    client = QdrantClient(location=":memory:")
    storage = QdrantStorageManager(client=client, model_name="BAAI/bge-base-en-v1.5", device="cpu")

    p1 = ParentSection(
        parent_id="td__car_rental__benefits", card_id="td_vip", card_name="TD VIP",
        policy_name="Car Rental Insurance", section_title="Benefits", page_start=24, page_end=25,
        content="Coverage covers rental cars up to $85,000 MSRP for up to 48 days."
    )
    p2 = ParentSection(
        parent_id="td__car_rental__exclusions", card_id="td_vip", card_name="TD VIP",
        policy_name="Car Rental Insurance", section_title="Exclusions", page_start=26, page_end=27,
        content="Excludes commercial vans, off-road vehicles, and exotic cars."
    )

    c1 = ChildChunk("td__c1", p1.parent_id, "td_vip", "TD VIP", "Car Rental", "Benefits", 24,
                    "MSRP limit is $85,000 for rental vehicles.", "[TD] (Page 24)", p1.content)
    c2 = ChildChunk("td__c2", p1.parent_id, "td_vip", "TD VIP", "Car Rental", "Benefits", 25,
                    "Rental duration limit is 48 consecutive days.", "[TD] (Page 25)", p1.content)
    c3 = ChildChunk("td__c3", p2.parent_id, "td_vip", "TD VIP", "Car Rental", "Exclusions", 26,
                    "Commercial vans and off-road driving are excluded.", "[TD] (Page 26)", p2.content)

    storage.save_card_chunks("td_vip", [p1, p2], [c1, c2, c3])

    retriever = HybridParentRetriever(storage_manager=storage)
    result = retriever.retrieve(card_id="td_vip", query="What is the maximum car rental MSRP?")

    assert isinstance(result, RetrievedContext)
    # Deduplication test: Even though c1 and c2 match, p1 should appear only once
    parent_ids = [p.parent_id for p in result.parents]
    assert len(parent_ids) == len(set(parent_ids))
    assert p1.parent_id in parent_ids

    # Citations test
    assert len(result.citations) >= 1
    assert result.citations[0]["page_number"] in [24, 25]
    assert "85,000" in result.citations[0]["content"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_retriever.py -v`  
Expected: FAIL with missing `HybridParentRetriever` or `RetrievedContext`.

- [ ] **Step 3: Implement `HybridParentRetriever` in `src/retriever.py`**

Refactor `src/retriever.py` to:
1. Define `RetrievedContext(parents: List[ParentSection], citations: List[Dict[str, Any]])`.
2. Generate dense query via `storage_manager.embedder.encode([query])`.
3. Generate sparse query via `storage_manager.bm25_vectorizer.encode_query(query)`.
4. Execute `client.query_points`:
   - `prefetch`: dense query (`using="dense"`) + sparse query (`using="bm25"`), both with `card_id` filter.
   - `query`: `models.FusionQuery(fusion=models.Fusion.RRF)`.
   - `limit`: 8 child points.
5. Deduplicate parents by `parent_id` while preserving rank order, extracting top 2–3 unique `ParentSection`s.
6. Assemble child citations (page number, breadcrumb, clause content, score).

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_retriever.py -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/retriever.py tests/test_retriever.py
git commit -m "feat(retriever): implement cloud hybrid retrieval with parent deduplication"
```

---

### Task 5: Generator Prompt Update & API SSE Streaming Integration

**Files:**
- Modify: `src/generator.py`
- Modify: `api/main.py`
- Modify: `tests/test_generator.py`

**Interfaces:**
- Consumes: `RetrievedContext`
- Produces: 
  - `LLMGenerator.generate_response(question, retrieved_context) -> GroundedResponse`
  - `LLMGenerator.stream_response(question, retrieved_context) -> Generator[str, None, None]`
  - FastAPI `/api/query` streaming Server-Sent Events with parent synthesis and child citations drawer data.

- [ ] **Step 1: Write the failing test for `LLMGenerator` with `RetrievedContext`**

```python
# tests/test_generator.py
from src.parser import ParentSection
from src.retriever import RetrievedContext
from src.generator import LLMGenerator

def test_generator_builds_parent_context_prompt():
    gen = LLMGenerator()
    parent = ParentSection(
        parent_id="p1", card_id="cibc_vip", card_name="CIBC VIP",
        policy_name="Flight Delay", section_title="Benefits", page_start=35, page_end=36,
        content="Flight delay pays up to $500 for hotel and food after 4 hours."
    )
    citation = {
        "page_number": 36, "breadcrumb": "[CIBC] > [Flight Delay] (Page 36)",
        "content": "Flight delay pays up to $500."
    }
    context = RetrievedContext(parents=[parent], citations=[citation])

    prompt = gen._build_context_prompt("How much do I get for delayed flight?", context)
    assert "--- Parent Section 1: [Flight Delay] > Benefits (Pages 35-36) ---" in prompt
    assert "Flight delay pays up to $500 for hotel and food" in prompt
    assert "USER QUESTION:" in prompt
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_generator.py -v`  
Expected: FAIL with signature or attribute error.

- [ ] **Step 3: Update `src/generator.py` and `api/main.py`**

1. Update `src/generator.py` to:
   - Accept `RetrievedContext` in `_build_context_prompt()`, `generate_response()`, and `stream_response()`.
   - Format parent sections with full headings and page spans.
   - Retain the strict 4-part schema instructions.
2. Update `api/main.py` to:
   - Call `HybridParentRetriever.retrieve(card_id, req.query)`.
   - Send `data: {"type": "citations", "citations": retrieved_context.citations}` over SSE.
   - Stream synthesized tokens from `generator.stream_response(req.query, retrieved_context)`.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_generator.py -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/generator.py api/main.py tests/test_generator.py
git commit -m "feat(generator): update LLM prompt and API SSE stream for parent-document retrieval"
```

---

### Task 6: Cloud Indexing Script, Benchmark Suite & Documentation

**Files:**
- Create: `.env.example`
- Create: `scripts/index_all_cards.py`
- Modify: `tests/test_benchmark_td.py`
- Create: `tests/test_benchmark_cibc.py`
- Modify: `docs/ARCHITECTURE_AND_TECHNICAL_REFERENCE.md`

**Interfaces:**
- Consumes: `data/` PDF files, Qdrant Cloud cluster
- Produces: 
  - Complete index in Qdrant Cloud for all credit cards.
  - Benchmark passing on CIBC and TD policies.
  - Updated architectural reference manual.

- [ ] **Step 1: Write `.env.example` and CIBC Benchmark Test**

```python
# tests/test_benchmark_cibc.py
import pytest
from src.storage import QdrantStorageManager
from src.retriever import HybridParentRetriever
from qdrant_client import QdrantClient

def test_cibc_benchmark_rental_car_msrp():
    # Tests that CIBC rental car query retrieves the $100,000 MSRP parent section and child citation
    client = QdrantClient(location=":memory:")
    storage = QdrantStorageManager(client=client, device="cpu")
    # Ingest CIBC PDF
    from src.parser import InsuranceDocParser
    parser = InsuranceDocParser()
    parents, children = parser.parse_pdf("data/cibc/CIBC Aeroplan Visa Infinite Privilege Card-Insurance.pdf",
                                         card_name="CIBC Aeroplan VIP", card_id="cibc_aeroplan_vip")
    storage.save_card_chunks("cibc_aeroplan_vip", parents, children)

    retriever = HybridParentRetriever(storage_manager=storage)
    ctx = retriever.retrieve("cibc_aeroplan_vip", "What is the maximum rental car MSRP coverage limit?")
    
    combined_parent_text = "\n".join(p.content for p in ctx.parents)
    assert "100,000" in combined_parent_text
    assert "48" in combined_parent_text
```

- [ ] **Step 2: Run benchmark test to verify it passes**

Run: `pytest tests/test_benchmark_cibc.py -v`  
Expected: PASS

- [ ] **Step 3: Implement `scripts/index_all_cards.py`**

Create automated CLI script that:
1. Loads `.env`
2. Scans `data/*/*.pdf` via `CardRegistry`
3. Parses each card into `ParentSection`s and `ChildChunk`s
4. Upserts into Qdrant Cloud with progress logs
5. Reports total parents, total children, and vector collection health.

- [ ] **Step 4: Run full test suite**

Run: `pytest tests/ -v`  
Expected: All unit, storage, retrieval, and benchmark tests PASS.

- [ ] **Step 5: Commit & update docs**

```bash
git add .env.example scripts/index_all_cards.py tests/test_benchmark_cibc.py tests/test_benchmark_td.py docs/ARCHITECTURE_AND_TECHNICAL_REFERENCE.md
git commit -m "feat: complete parent-document retrieval pipeline with cloud qdrant benchmarks"
```
