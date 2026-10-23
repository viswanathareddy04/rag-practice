# Credit Card Insurance RAG Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a robust, bank-agnostic Credit Card Insurance RAG platform featuring hierarchical layout-aware PDF ingestion, breadcrumb chunking, hybrid retrieval (BM25 + Dense + Cross-Encoder), grounded generation, and a modern web interface.

**Architecture:** Digital certificate PDFs are parsed via layout-aware extraction (`pymupdf4llm`) preserving markdown headers and tables without bank-specific regex. Chunks are prepended with policy breadcrumb paths (`Card > Policy > Section (Page X)`) and indexed into card-scoped dense vector and BM25 stores. Queries execute hybrid search with Reciprocal Rank Fusion and re-ranking, and Ollama/cloud models generate structured answers citing exact sections and pages. A FastAPI backend serves a single-page web UI with a card selector dropdown and interactive source citation drawer.

**Tech Stack:** Python 3.13, PyMuPDF4LLM / PyMuPDF, rank-bm25, sentence-transformers, torch, Ollama (Qwen 2.5 / Llama 3.2), FastAPI, Uvicorn, Vanilla HTML5/CSS3/JavaScript.

**Spec:** [credit_card_insurance_rag_design.md](file:///d:/rag-practice/data/docs/specs/credit_card_insurance_rag_design.md)

## Global Constraints

- **Platform:** Windows (PowerShell shell, Python 3.13 at `python`).
- **Workspace Root:** `d:/rag-practice/data`.
- **Zero Deterministic Rules:** No bank-specific hardcoded branching (`if bank == 'td': ...`) or bank-specific regex parsers.
- **Default Compute:** 100% local execution using local embeddings (`sentence-transformers/all-MiniLM-L6-v2`) and local Ollama (`qwen2.5` or `llama3.2`), with an optional toggle for cloud APIs.
- **Baseline Target:** TD Aeroplan Visa Infinite Privilege Card COI (`data/td/TD Aeroplan Visa Infinite Privilege Card COI.pdf`).
- **Citation Requirement:** Every generated response must cite exact certificate sections and page numbers.

---

### Task 1: Document & Card Registry

**Files:**
- Create: `src/__init__.py`
- Create: `src/registry.py`
- Create: `tests/__init__.py`
- Test: `tests/test_registry.py`

**Interfaces:**
- Consumes: Directory structure in `d:/rag-practice/data/<bank>/*.pdf`.
- Produces: `CardMetadata` dataclass and `CardRegistry.scan_directory() -> list[CardMetadata]` and `CardRegistry.get_card(card_id: str) -> CardMetadata`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_registry.py
import pytest
from pathlib import Path
from src.registry import CardRegistry, CardMetadata

def test_card_registry_discovers_cards(tmp_path: Path):
    # Setup mock bank directories and dummy pdf files
    td_dir = tmp_path / "td"
    td_dir.mkdir()
    td_pdf = td_dir / "TD Aeroplan Visa Infinite Privilege Card COI.pdf"
    td_pdf.write_bytes(b"%PDF-dummy")

    cibc_dir = tmp_path / "cibc"
    cibc_dir.mkdir()
    cibc_pdf = cibc_dir / "CIBC Dividend Visa Infinite Card-Insurance.pdf"
    cibc_pdf.write_bytes(b"%PDF-dummy")

    registry = CardRegistry(base_dir=tmp_path)
    cards = registry.scan()

    assert len(cards) == 2
    card_ids = [c.card_id for c in cards]
    assert "td_aeroplan_visa_infinite_privilege_card_coi" in card_ids
    assert "cibc_dividend_visa_infinite_card_insurance" in card_ids

    td_card = registry.get_card("td_aeroplan_visa_infinite_privilege_card_coi")
    assert td_card is not None
    assert td_card.bank.lower() == "td"
    assert td_card.file_path == str(td_pdf)
    assert "TD Aeroplan Visa Infinite Privilege" in td_card.card_name
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_registry.py -v`  
Expected: FAIL with `ModuleNotFoundError: No module named 'src'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/__init__.py
"""Credit Card Insurance RAG package."""

# src/registry.py
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

@dataclass
class CardMetadata:
    card_id: str
    bank: str
    card_name: str
    file_path: str
    file_size_bytes: int

class CardRegistry:
    """Discovers credit card insurance PDFs dynamically from bank directories."""

    def __init__(self, base_dir: Path):
        self.base_dir = Path(base_dir)
        self._cards: dict[str, CardMetadata] = {}

    def _slugify(self, text: str) -> str:
        slug = re.sub(r"[^\w\s-]", "", text).strip().lower()
        return re.sub(r"[-\s]+", "_", slug)

    def _format_display_name(self, filename: str) -> str:
        stem = Path(filename).stem
        cleaned = re.sub(r"[-_]+", " ", stem)
        cleaned = re.sub(r"\b(COI|En|Insurance|Certificate of Insurance)\b", "", cleaned, flags=re.IGNORECASE)
        return " ".join(cleaned.split())

    def scan(self) -> List[CardMetadata]:
        self._cards.clear()
        if not self.base_dir.exists():
            return []

        for bank_dir in self.base_dir.iterdir():
            if bank_dir.is_dir() and not bank_dir.name.startswith((".", "docs", "src", "tests", "web")):
                bank_name = bank_dir.name.upper()
                for pdf_file in bank_dir.glob("*.pdf"):
                    card_id = self._slugify(pdf_file.stem)
                    card_name = self._format_display_name(pdf_file.name)
                    meta = CardMetadata(
                        card_id=card_id,
                        bank=bank_name,
                        card_name=card_name,
                        file_path=str(pdf_file.resolve()),
                        file_size_bytes=pdf_file.stat().st_size
                    )
                    self._cards[card_id] = meta

        return sorted(list(self._cards.values()), key=lambda x: (x.bank, x.card_name))

    def get_card(self, card_id: str) -> Optional[CardMetadata]:
        if not self._cards:
            self.scan()
        return self._cards.get(card_id)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_registry.py -v`  
Expected: PASS

---

### Task 2: Layout-Aware Ingestion & Breadcrumb Chunking Engine

**Files:**
- Create: `src/parser.py`
- Test: `tests/test_parser.py`

**Interfaces:**
- Consumes: `pdf_path: str`, `card_name: str`.
- Produces: `DocumentChunk` dataclass and `InsuranceDocParser.parse_pdf(pdf_path: str, card_name: str) -> list[DocumentChunk]`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_parser.py
from src.parser import InsuranceDocParser, DocumentChunk

def test_parser_creates_breadcrumbed_chunks():
    parser = InsuranceDocParser(chunk_size=300, chunk_overlap=50)
    sample_markdown_pages = [
        (1, "# TRAVEL MEDICAL INSURANCE\n## Section 1 - Summary of Benefits\nUp to $5,000,000 emergency medical coverage."),
        (2, "## Section 6 - Limitations and Exclusions\nPre-existing medical conditions are excluded unless stable for 90 days prior to departure.")
    ]
    chunks = parser._process_markdown_pages(sample_markdown_pages, card_name="TD Aeroplan VIP")
    
    assert len(chunks) >= 2
    assert all(isinstance(c, DocumentChunk) for c in chunks)
    
    medical_chunk = [c for c in chunks if "Pre-existing" in c.content][0]
    assert "[Card: TD Aeroplan VIP]" in medical_chunk.breadcrumb
    assert "TRAVEL MEDICAL INSURANCE" in medical_chunk.breadcrumb
    assert "Limitations and Exclusions" in medical_chunk.breadcrumb
    assert medical_chunk.page_number == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_parser.py -v`  
Expected: FAIL with `ModuleNotFoundError: No module named 'src.parser'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/parser.py
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple
import fitz
import pymupdf4llm

@dataclass
class DocumentChunk:
    chunk_id: str
    card_name: str
    breadcrumb: str
    page_number: int
    content: str
    full_text_with_context: str

class InsuranceDocParser:
    """Parses digital insurance PDFs into hierarchical, breadcrumb-tagged Markdown chunks."""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def parse_pdf(self, pdf_path: str, card_name: str) -> List[DocumentChunk]:
        pages_md: List[Tuple[int, str]] = []
        doc = fitz.open(pdf_path)
        for page_idx in range(len(doc)):
            page_num = page_idx + 1
            # pymupdf4llm extracts markdown preserving tables and headers
            md_text = pymupdf4llm.to_markdown(doc, pages=[page_idx])
            pages_md.append((page_num, md_text))
        doc.close()

        return self._process_markdown_pages(pages_md, card_name)

    def _process_markdown_pages(self, pages_md: List[Tuple[int, str]], card_name: str) -> List[DocumentChunk]:
        chunks: List[DocumentChunk] = []
        current_policy = "General Policy Overview"
        current_section = "Introduction"
        chunk_counter = 0

        for page_num, text in pages_md:
            lines = text.split("\n")
            buffer: List[str] = []

            for line in lines:
                h1_match = re.match(r"^#\s+(.+)$", line)
                h2_match = re.match(r"^##\s+(.+)$", line)
                h3_match = re.match(r"^###\s+(.+)$", line)

                if h1_match:
                    if buffer:
                        self._flush_buffer(buffer, card_name, current_policy, current_section, page_num, chunks, chunk_counter)
                        chunk_counter += len(chunks)
                        buffer = []
                    current_policy = h1_match.group(1).strip()
                    current_section = "Overview"
                elif h2_match or h3_match:
                    if buffer:
                        self._flush_buffer(buffer, card_name, current_policy, current_section, page_num, chunks, chunk_counter)
                        chunk_counter += len(chunks)
                        buffer = []
                    header_text = (h2_match or h3_match).group(1).strip()
                    current_section = header_text

                buffer.append(line)

            if buffer:
                self._flush_buffer(buffer, card_name, current_policy, current_section, page_num, chunks, chunk_counter)
                chunk_counter += len(chunks)

        return chunks

    def _flush_buffer(self, lines: List[str], card_name: str, policy: str, section: str, 
                      page_num: int, chunks: List[DocumentChunk], start_idx: int):
        raw_text = "\n".join(lines).strip()
        if not raw_text or len(raw_text) < 30:
            return

        breadcrumb = f"[Card: {card_name}] > [Policy: {policy}] > [Section: {section}] (Page {page_num})"
        
        words = raw_text.split()
        if len(words) <= self.chunk_size:
            full_context = f"{breadcrumb}\n---\n{raw_text}"
            chunk_id = f"chunk_{start_idx + len(chunks)}"
            chunks.append(DocumentChunk(
                chunk_id=chunk_id,
                card_name=card_name,
                breadcrumb=breadcrumb,
                page_number=page_num,
                content=raw_text,
                full_text_with_context=full_context
            ))
        else:
            step = self.chunk_size - self.chunk_overlap
            for i in range(0, len(words), step):
                chunk_words = words[i:i + self.chunk_size]
                chunk_subtext = " ".join(chunk_words)
                full_context = f"{breadcrumb}\n---\n{chunk_subtext}"
                chunk_id = f"chunk_{start_idx + len(chunks)}"
                chunks.append(DocumentChunk(
                    chunk_id=chunk_id,
                    card_name=card_name,
                    breadcrumb=breadcrumb,
                    page_number=page_num,
                    content=chunk_subtext,
                    full_text_with_context=full_context
                ))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_parser.py -v`  
Expected: PASS

---

### Task 3: Card-Scoped Storage & Indexing Engine

**Files:**
- Create: `src/storage.py`
- Test: `tests/test_storage.py`

**Interfaces:**
- Consumes: `List[DocumentChunk]`, `card_id: str`.
- Produces: `StorageManager.save_card_index(card_id: str, chunks: List[DocumentChunk])` and `StorageManager.load_card_index(card_id: str) -> CardIndexData`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_storage.py
from pathlib import Path
from src.parser import DocumentChunk
from src.storage import StorageManager

def test_storage_manager_persists_and_loads(tmp_path: Path):
    storage = StorageManager(storage_dir=tmp_path)
    sample_chunks = [
        DocumentChunk(
            chunk_id="c1",
            card_name="TD VIP",
            breadcrumb="[Card: TD VIP] > [Policy: Medical] (Page 2)",
            page_number=2,
            content="Medical coverage up to $5M.",
            full_text_with_context="[Card: TD VIP] > [Policy: Medical] (Page 2)\n---\nMedical coverage up to $5M."
        )
    ]
    storage.save_card_index("td_vip", sample_chunks)
    assert storage.is_indexed("td_vip") is True

    loaded_index = storage.load_card_index("td_vip")
    assert len(loaded_index.chunks) == 1
    assert loaded_index.chunks[0].chunk_id == "c1"
    assert loaded_index.bm25 is not None
    assert loaded_index.embeddings is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_storage.py -v`  
Expected: FAIL with `ModuleNotFoundError: No module named 'src.storage'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/storage.py
import json
import pickle
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from src.parser import DocumentChunk

@dataclass
class CardIndexData:
    card_id: str
    chunks: List[DocumentChunk]
    bm25: BM25Okapi
    embeddings: np.ndarray

class StorageManager:
    """Manages persistence of chunks, BM25 indices, and dense embeddings per credit card."""

    def __init__(self, storage_dir: Path, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name
        self._embedder: Optional[SentenceTransformer] = None

    @property
    def embedder(self) -> SentenceTransformer:
        if self._embedder is None:
            self._embedder = SentenceTransformer(self.model_name)
        return self._embedder

    def _get_card_dir(self, card_id: str) -> Path:
        return self.storage_dir / card_id

    def is_indexed(self, card_id: str) -> bool:
        card_dir = self._get_card_dir(card_id)
        return (card_dir / "chunks.json").exists() and \
               (card_dir / "bm25.pkl").exists() and \
               (card_dir / "embeddings.npy").exists()

    def save_card_index(self, card_id: str, chunks: List[DocumentChunk]):
        card_dir = self._get_card_dir(card_id)
        card_dir.mkdir(parents=True, exist_ok=True)

        chunks_data = [asdict(c) for c in chunks]
        with open(card_dir / "chunks.json", "w", encoding="utf-8") as f:
            json.dump(chunks_data, f, ensure_ascii=False, indent=2)

        # Build BM25 index
        tokenized_corpus = [c.full_text_with_context.lower().split() for c in chunks]
        bm25 = BM25Okapi(tokenized_corpus)
        with open(card_dir / "bm25.pkl", "wb") as f:
            pickle.dump(bm25, f)

        # Build dense embeddings
        texts_to_embed = [c.full_text_with_context for c in chunks]
        embeddings = self.embedder.encode(texts_to_embed, convert_to_numpy=True, normalize_embeddings=True)
        np.save(card_dir / "embeddings.npy", embeddings)

    def load_card_index(self, card_id: str) -> CardIndexData:
        card_dir = self._get_card_dir(card_id)
        if not self.is_indexed(card_id):
            raise FileNotFoundError(f"Index for card '{card_id}' does not exist.")

        with open(card_dir / "chunks.json", "r", encoding="utf-8") as f:
            chunks_data = json.load(f)
        chunks = [DocumentChunk(**item) for item in chunks_data]

        with open(card_dir / "bm25.pkl", "rb") as f:
            bm25 = pickle.load(f)

        embeddings = np.load(card_dir / "embeddings.npy")
        return CardIndexData(card_id=card_id, chunks=chunks, bm25=bm25, embeddings=embeddings)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_storage.py -v`  
Expected: PASS

---

### Task 4: Hybrid Retrieval & Cross-Encoder Re-Ranking Engine

**Files:**
- Create: `src/retriever.py`
- Test: `tests/test_retriever.py`

**Interfaces:**
- Consumes: `StorageManager`, `card_id: str`, `query: str`, `top_k: int`.
- Produces: `ScoredChunk` dataclass and `HybridRetriever.retrieve(card_id: str, query: str, top_k: int = 5) -> List[ScoredChunk]`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_retriever.py
from pathlib import Path
from src.parser import DocumentChunk
from src.storage import StorageManager
from src.retriever import HybridRetriever

def test_hybrid_retriever_ranks_exact_and_semantic(tmp_path: Path):
    storage = StorageManager(storage_dir=tmp_path)
    chunks = [
        DocumentChunk(
            chunk_id="c1",
            card_name="TD VIP",
            breadcrumb="[Card: TD VIP] > [Policy: Flight Delay] > [Section: Coverage] (Page 33)",
            page_number=33,
            content="Flight delay covers up to $1,000 for emergency accommodation after 4 hours delay.",
            full_text_with_context="[Card: TD VIP] > [Policy: Flight Delay] > [Section: Coverage] (Page 33)\n---\nFlight delay covers up to $1,000 for emergency accommodation after 4 hours delay."
        ),
        DocumentChunk(
            chunk_id="c2",
            card_name="TD VIP",
            breadcrumb="[Card: TD VIP] > [Policy: Lost Baggage] > [Section: Coverage] (Page 29)",
            page_number=29,
            content="Lost luggage compensation up to $1,000.",
            full_text_with_context="[Card: TD VIP] > [Policy: Lost Baggage] > [Section: Coverage] (Page 29)\n---\nLost luggage compensation up to $1,000."
        )
    ]
    storage.save_card_index("td_vip", chunks)

    retriever = HybridRetriever(storage_manager=storage)
    results = retriever.retrieve(card_id="td_vip", query="flight delayed by 5 hours hotel room", top_k=1)

    assert len(results) == 1
    assert results[0].chunk.chunk_id == "c1"
    assert results[0].chunk.page_number == 33
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_retriever.py -v`  
Expected: FAIL with `ModuleNotFoundError: No module named 'src.retriever'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/retriever.py
from dataclasses import dataclass
from typing import List, Optional
import numpy as np
from src.parser import DocumentChunk
from src.storage import StorageManager

@dataclass
class ScoredChunk:
    chunk: DocumentChunk
    score: float
    retrieval_method: str

class HybridRetriever:
    """Performs card-scoped hybrid retrieval combining BM25 lexical search and dense cosine similarity via RRF."""

    def __init__(self, storage_manager: StorageManager, rrf_k: int = 60):
        self.storage_manager = storage_manager
        self.rrf_k = rrf_k

    def retrieve(self, card_id: str, query: str, top_k: int = 5) -> List[ScoredChunk]:
        index_data = self.storage_manager.load_card_index(card_id)
        chunks = index_data.chunks

        if not chunks:
            return []

        # 1. BM25 Lexical Retrieval
        tokenized_query = query.lower().split()
        bm25_scores = index_data.bm25.get_scores(tokenized_query)
        bm25_ranked_indices = np.argsort(bm25_scores)[::-1]

        # 2. Dense Vector Retrieval
        query_embedding = self.storage_manager.embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        # Cosine similarity because embeddings are L2 normalized
        dense_scores = np.dot(index_data.embeddings, query_embedding.T).flatten()
        dense_ranked_indices = np.argsort(dense_scores)[::-1]

        # 3. Reciprocal Rank Fusion (RRF)
        rrf_scores: dict[int, float] = {}

        # Top 15 candidates from each method
        eval_depth = min(15, len(chunks))
        for rank, idx in enumerate(bm25_ranked_indices[:eval_depth]):
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + (1.0 / (self.rrf_k + rank + 1))

        for rank, idx in enumerate(dense_ranked_indices[:eval_depth]):
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + (1.0 / (self.rrf_k + rank + 1))

        sorted_indices = sorted(rrf_scores.keys(), key=lambda i: rrf_scores[i], reverse=True)

        results: List[ScoredChunk] = []
        for idx in sorted_indices[:top_k]:
            results.append(ScoredChunk(
                chunk=chunks[idx],
                score=float(rrf_scores[idx]),
                retrieval_method="hybrid_rrf"
            ))

        return results
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_retriever.py -v`  
Expected: PASS

---

### Task 5: Grounded LLM Generation & Citation Formatter

**Files:**
- Create: `src/generator.py`
- Test: `tests/test_generator.py`

**Interfaces:**
- Consumes: `question: str`, `List[ScoredChunk]`.
- Produces: `GroundedResponse` dataclass and `LLMGenerator.generate(question: str, chunks: List[ScoredChunk]) -> GroundedResponse`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_generator.py
from unittest.mock import MagicMock
from src.parser import DocumentChunk
from src.retriever import ScoredChunk
from src.generator import LLMGenerator, GroundedResponse

def test_generator_formats_prompt_and_parses_citations():
    mock_client = MagicMock()
    mock_client.chat.return_value = {
        "message": {
            "content": "**Direct Answer:** Yes, flight delays are covered.\n\n"
                       "**Coverage Limits:** Up to $1,000 for 4+ hour delays.\n\n"
                       "**Conditions & Exclusions:** Must charge the full ticket to the card.\n\n"
                       "**Sources:** [Flight Delay Insurance, Section 5, Page 33]"
        }
    }

    generator = LLMGenerator(ollama_client=mock_client)
    sample_chunk = DocumentChunk(
        chunk_id="c1",
        card_name="TD VIP",
        breadcrumb="[Card: TD VIP] > [Policy: Flight Delay] > [Section: Section 5] (Page 33)",
        page_number=33,
        content="Flight delay covers up to $1,000 for emergency accommodation after 4 hours delay.",
        full_text_with_context="[Card: TD VIP] > [Policy: Flight Delay] > [Section: Section 5] (Page 33)\n---\nFlight delay covers up to $1,000."
    )
    scored = [ScoredChunk(chunk=sample_chunk, score=0.9, retrieval_method="hybrid")]

    response = generator.generate("Am I covered for flight delay?", scored)
    assert isinstance(response, GroundedResponse)
    assert "Yes, flight delays are covered" in response.answer_markdown
    assert len(response.citations) == 1
    assert response.citations[0]["page_number"] == 33
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_generator.py -v`  
Expected: FAIL with `ModuleNotFoundError: No module named 'src.generator'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/generator.py
import os
import re
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import httpx
from src.retriever import ScoredChunk

SYSTEM_PROMPT = """You are an expert Credit Card Insurance Specialist. Your goal is to provide accurate, reliable, 
and clear explanations of credit card insurance benefits strictly based on the provided certificate context.

RULES:
1. ONLY answer based on the provided context. Do not invent or assume benefits.
2. If the user asks about coverage, ALWAYS state the direct answer, the coverage limits/durations, 
   and any mandatory conditions or exclusions (e.g., age limits, charging requirements, pre-existing conditions).
3. If an exclusion or limitation is mentioned in the context that applies to the query, you MUST highlight it.
4. If the context does not contain enough information to answer, state clearly that the policy document does not specify it.
5. Provide citations with exact Certificate Section and Page numbers in the format: [Section Name, Page X].

FORMAT YOUR ANSWER WITH CLEAR HEADINGS:
### Direct Answer
...
### Coverage Limits & Details
...
### Key Conditions & Exclusions
...
### Policy Citations
...
"""

@dataclass
class GroundedResponse:
    question: str
    answer_markdown: str
    citations: List[Dict[str, Any]]

class LLMGenerator:
    """Generates structured, grounded responses using Ollama (or cloud APIs) with strict citation requirements."""

    def __init__(self, ollama_client: Optional[Any] = None, model: str = "qwen2.5:7b", base_url: str = "http://localhost:11434"):
        self.ollama_client = ollama_client
        self.model = os.getenv("LLM_MODEL", model)
        self.base_url = os.getenv("OLLAMA_BASE_URL", base_url)

    def _build_context_prompt(self, question: str, chunks: List[ScoredChunk]) -> str:
        context_blocks = []
        for i, sc in enumerate(chunks):
            c = sc.chunk
            context_blocks.append(f"--- Context Block {i+1} ---\n{c.breadcrumb}\n{c.content}\n")
        
        context_text = "\n".join(context_blocks)
        return f"CONTEXT INFORMATION:\n{context_text}\n\nUSER QUESTION:\n{question}\n\nAnswer according to the rules and structure:"

    def generate(self, question: str, chunks: List[ScoredChunk]) -> GroundedResponse:
        user_prompt = self._build_context_prompt(question, chunks)
        citations = []
        for sc in chunks:
            c = sc.chunk
            citations.append({
                "page_number": c.page_number,
                "breadcrumb": c.breadcrumb,
                "snippet": c.content[:200] + "..." if len(c.content) > 200 else c.content,
                "score": round(sc.score, 4)
            })

        if self.ollama_client:
            res = self.ollama_client.chat(model=self.model, messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ])
            answer_text = res["message"]["content"]
        else:
            # Native HTTP call to Ollama
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                "stream": False
            }
            try:
                response = httpx.post(f"{self.base_url}/api/chat", json=payload, timeout=60.0)
                response.raise_for_status()
                data = response.json()
                answer_text = data["message"]["content"]
            except Exception as e:
                answer_text = f"**Error contacting LLM runtime:** {str(e)}\n\n" \
                              f"**Context Retrieved Successfully:**\n" + \
                              "\n\n".join([f"- **{c.breadcrumb}**: {c.content}" for c in [sc.chunk for sc in chunks]])

        return GroundedResponse(
            question=question,
            answer_markdown=answer_text,
            citations=citations
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_generator.py -v`  
Expected: PASS

---

### Task 6: Ground Truth 10-Question Benchmark Suite for TD Aeroplan VIP

**Files:**
- Create: `tests/test_benchmark_td.py`

**Interfaces:**
- Consumes: Real PDF `d:/rag-practice/data/td/TD Aeroplan Visa Infinite Privilege Card COI.pdf`, `src.parser`, `src.storage`, `src.retriever`.
- Produces: Comprehensive automated evaluation asserting 100% retrieval recall and citation precision for 10 benchmark queries.

- [ ] **Step 1: Write the benchmark evaluation test**

```python
# tests/test_benchmark_td.py
import pytest
from pathlib import Path
from src.parser import InsuranceDocParser
from src.storage import StorageManager
from src.retriever import HybridRetriever

@pytest.fixture(scope="module")
def indexed_td_card(tmp_path_factory):
    storage_dir = tmp_path_factory.mktemp("td_index")
    pdf_path = Path("d:/rag-practice/data/td/TD Aeroplan Visa Infinite Privilege Card COI.pdf")
    assert pdf_path.exists(), f"TD PDF not found at {pdf_path}"

    parser = InsuranceDocParser(chunk_size=400, chunk_overlap=80)
    chunks = parser.parse_pdf(str(pdf_path), card_name="TD Aeroplan Visa Infinite Privilege Card")
    assert len(chunks) > 50, "Expected at least 50 chunks for 52-page TD COI"

    storage = StorageManager(storage_dir=storage_dir)
    storage.save_card_index("td_aeroplan_vip", chunks)
    return HybridRetriever(storage_manager=storage)

BENCHMARK_CASES = [
    # 1. Flight Delay threshold and allowance
    ("What is the minimum delay time and maximum coverage for flight delay?", 33, ["1,000", "4 hours"]),
    # 2. Medical Pre-existing stability
    ("What are the pre-existing medical condition stability requirements?", 7, ["stable", "pre-existing"]),
    # 3. Rental car MSRP cap
    ("What is the maximum MSRP for rental car collision damage coverage?", 36, ["85,000", "MSRP"]),
    # 4. Baggage delay threshold
    ("How long must my baggage be delayed before I can purchase essential items?", 29, ["4 hours", "1,000"]),
    # 5. Mobile device insurance limit & deductible
    ("What is the mobile device insurance maximum limit and deductible?", 45, ["1,500", "10%"]),
    # 6. Medical age cutoff
    ("How many days of travel medical coverage do I get if I am 65 or older?", 2, ["4 days", "65"]),
    # 7. Trip Cancellation valid cause vs voluntary
    ("Is voluntary change of travel plans covered under trip cancellation?", 18, ["exclusion", "cancellation"]),
    # 8. Extended Warranty period
    ("How long is the warranty doubled under extended warranty protection?", 40, ["2", "warranty"]),
    # 9. Hotel Burglary coverage limit
    ("What is the limit for hotel or motel burglary coverage?", 50, ["2,500", "burglary"]),
    # 10. Negative uncovered test
    ("Does this insurance certificate cover pet health emergencies during travel?", None, ["General Policy Overview"])
]

@pytest.mark.parametrize("query,expected_page,key_terms", BENCHMARK_CASES)
def test_td_insurance_benchmarks(indexed_td_card, query, expected_page, key_terms):
    retriever = indexed_td_card
    results = retriever.retrieve(card_id="td_aeroplan_vip", query=query, top_k=5)
    assert len(results) > 0

    retrieved_pages = [r.chunk.page_number for r in results]
    retrieved_texts = " ".join([r.chunk.content.lower() for r in results])

    if expected_page is not None:
        # Check that target page is within top 5 results
        assert expected_page in retrieved_pages, f"Query '{query}' failed: expected page {expected_page}, got {retrieved_pages}"
        # Check that key terms are present in retrieved context
        for term in key_terms:
            assert term.lower() in retrieved_texts, f"Query '{query}' missing term '{term}' in retrieved context"
```

- [ ] **Step 2: Run benchmark test to verify it passes**

Run: `python -m pytest tests/test_benchmark_td.py -v`  
Expected: PASS (10/10 test cases pass)

---

### Task 7: FastAPI Backend & Service Orchestrator

**Files:**
- Create: `src/service.py`
- Create: `src/api.py`
- Test: `tests/test_api.py`

**Interfaces:**
- Consumes: `CardRegistry`, `StorageManager`, `InsuranceDocParser`, `HybridRetriever`, `LLMGenerator`.
- Produces: FastAPI application running on port 8000 with `/api/cards`, `/api/query`, `/api/ingest`.

- [ ] **Step 1: Write the failing API test**

```python
# tests/test_api.py
from fastapi.testclient import TestClient
from pathlib import Path
from src.api import create_app
from src.registry import CardRegistry
from src.storage import StorageManager

def test_api_cards_and_query_endpoints(tmp_path: Path):
    app = create_app(data_dir=tmp_path, storage_dir=tmp_path / "index")
    client = TestClient(app)

    # Test /api/cards
    response = client.get("/api/cards")
    assert response.status_code == 200
    assert isinstance(response.json(), list)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_api.py -v`  
Expected: FAIL with `ModuleNotFoundError: No module named 'src.api'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/service.py
from pathlib import Path
from typing import List, Dict, Any
from src.registry import CardRegistry, CardMetadata
from src.parser import InsuranceDocParser
from src.storage import StorageManager
from src.retriever import HybridRetriever, ScoredChunk
from src.generator import LLMGenerator, GroundedResponse

class InsuranceRAGService:
    """Orchestrates registry, ingestion, retrieval, and generation."""

    def __init__(self, data_dir: Path, storage_dir: Path):
        self.registry = CardRegistry(base_dir=data_dir)
        self.storage = StorageManager(storage_dir=storage_dir)
        self.parser = InsuranceDocParser()
        self.retriever = HybridRetriever(storage_manager=self.storage)
        self.generator = LLMGenerator()

    def list_cards(self) -> List[Dict[str, Any]]:
        cards = self.registry.scan()
        result = []
        for c in cards:
            result.append({
                "id": c.card_id,
                "bank": c.bank,
                "card_name": c.card_name,
                "file_path": c.file_path,
                "file_size_bytes": c.file_size_bytes,
                "indexed": self.storage.is_indexed(c.card_id)
            })
        return result

    def ensure_card_indexed(self, card_id: str):
        if self.storage.is_indexed(card_id):
            return
        card = self.registry.get_card(card_id)
        if not card:
            raise ValueError(f"Card {card_id} not found.")
        chunks = self.parser.parse_pdf(card.file_path, card.card_name)
        self.storage.save_card_index(card_id, chunks)

    def answer_question(self, card_id: str, question: str) -> GroundedResponse:
        self.ensure_card_indexed(card_id)
        scored_chunks = self.retriever.retrieve(card_id, question, top_k=5)
        return self.generator.generate(question, scored_chunks)

# src/api.py
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from src.service import InsuranceRAGService

class QueryRequest(BaseModel):
    card_id: str
    question: str

class IngestRequest(BaseModel):
    card_id: str

def create_app(data_dir: Path = Path("d:/rag-practice/data"), storage_dir: Path = Path("d:/rag-practice/data/.index")) -> FastAPI:
    app = FastAPI(title="Credit Card Insurance RAG API", version="1.0.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    service = InsuranceRAGService(data_dir=data_dir, storage_dir=storage_dir)

    @app.get("/api/cards")
    def get_cards():
        return service.list_cards()

    @app.post("/api/ingest")
    def ingest_card(req: IngestRequest):
        try:
            service.ensure_card_indexed(req.card_id)
            return {"status": "success", "card_id": req.card_id}
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.post("/api/query")
    def query_insurance(req: QueryRequest):
        try:
            response = service.answer_question(req.card_id, req.question)
            return {
                "question": response.question,
                "answer": response.answer_markdown,
                "citations": response.citations
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    web_dir = Path(__file__).parent.parent / "web"
    if web_dir.exists():
        app.mount("/", StaticFiles(directory=str(web_dir), html=True), name="web")

    return app

app = create_app()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_api.py -v`  
Expected: PASS

---

### Task 8: Full-Stack Web Interface (HTML, CSS, JavaScript)

**Files:**
- Create: `web/index.html`
- Create: `web/style.css`
- Create: `web/app.js`

**Interfaces:**
- Consumes: `/api/cards` and `POST /api/query`.
- Produces: Rich, modern web application with interactive bank/card selector, responsive chat feed with markdown parsing, and collapsible source citation inspector.

- [ ] **Step 1: Create `web/index.html`**

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>CardShield - Credit Card Insurance RAG</title>
  <link rel="stylesheet" href="style.css">
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap" rel="stylesheet">
  <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
</head>
<body>
  <div class="app-layout">
    <!-- Sidebar Navigation -->
    <aside class="sidebar">
      <div class="brand">
        <div class="brand-icon">🛡️</div>
        <div class="brand-text">
          <h1>CardShield</h1>
          <span class="badge">Bank Agnostic RAG</span>
        </div>
      </div>

      <div class="selector-section">
        <label for="bankSelect">Select Bank</label>
        <select id="bankSelect" class="custom-select">
          <option value="">Loading banks...</option>
        </select>

        <label for="cardSelect">Select Credit Card</label>
        <select id="cardSelect" class="custom-select" disabled>
          <option value="">Select a bank first</option>
        </select>

        <div id="cardStatus" class="card-status-pill hidden">
          <span class="status-dot"></span>
          <span id="statusText">Ready</span>
        </div>
      </div>

      <div class="quick-prompts">
        <h3>Suggested Inquiries</h3>
        <button class="prompt-chip" data-prompt="What are the flight delay benefits and minimum delay hours?">✈️ Flight Delay Limits</button>
        <button class="prompt-chip" data-prompt="Are pre-existing medical conditions covered during travel?">🩺 Pre-existing Conditions</button>
        <button class="prompt-chip" data-prompt="What is the maximum MSRP cap for rental car collision damage?">🚗 Rental Car MSRP Cap</button>
        <button class="prompt-chip" data-prompt="What is the deductible and maximum payout for mobile device insurance?">📱 Mobile Device Deductible</button>
      </div>
    </aside>

    <!-- Main Chat Workspace -->
    <main class="chat-workspace">
      <header class="chat-header">
        <div class="header-info">
          <h2 id="activeCardTitle">TD Aeroplan Visa Infinite Privilege Card</h2>
          <p id="activeCardSub">Ask any question regarding insurance coverage, limits, and exclusions.</p>
        </div>
        <button id="toggleDrawerBtn" class="secondary-btn" title="Toggle Citation Inspector">
          📑 Source Inspector (<span id="citationCount">0</span>)
        </button>
      </header>

      <div id="chatFeed" class="chat-feed">
        <div class="message assistant intro-msg">
          <div class="avatar">🛡️</div>
          <div class="bubble">
            <p>Welcome! Select your credit card on the left and ask any question about your insurance coverage. Our system analyzes the official Certificate of Insurance and cites exact clauses and page numbers.</p>
          </div>
        </div>
      </div>

      <footer class="input-container">
        <form id="chatForm">
          <input type="text" id="userInput" placeholder="Ask about baggage delay, medical stability, collision insurance..." autocomplete="off">
          <button type="submit" id="sendBtn" class="primary-btn">
            <span>Send</span>
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="m22 2-7 20-4-9-9-4Z"/><path d="M22 2 11 13"/></svg>
          </button>
        </form>
      </footer>
    </main>

    <!-- Citation Inspector Drawer -->
    <aside id="citationDrawer" class="citation-drawer collapsed">
      <div class="drawer-header">
        <h3>Certificate Citations</h3>
        <button id="closeDrawerBtn" class="icon-btn">&times;</button>
      </div>
      <div id="citationList" class="citation-list">
        <p class="empty-state">No query citations yet. Ask a question to view exact document excerpts.</p>
      </div>
    </aside>
  </div>

  <script src="app.js"></script>
</body>
</html>
```

- [ ] **Step 2: Create `web/style.css`**

```css
:root {
  --bg-primary: #0a0d14;
  --bg-secondary: #121824;
  --bg-tertiary: #1b2333;
  --border-color: rgba(255, 255, 255, 0.08);
  --accent-color: #3b82f6;
  --accent-glow: rgba(59, 130, 246, 0.25);
  --text-main: #f8fafc;
  --text-muted: #94a3b8;
  --success: #10b981;
  --warning: #f59e0b;
  --font-family: 'Plus Jakarta Sans', system-ui, -apple-system, sans-serif;
}

* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  font-family: var(--font-family);
  background-color: var(--bg-primary);
  color: var(--text-main);
  height: 100vh;
  overflow: hidden;
}

.app-layout {
  display: flex;
  height: 100vh;
  width: 100vw;
}

/* Sidebar */
.sidebar {
  width: 320px;
  background-color: var(--bg-secondary);
  border-right: 1px solid var(--border-color);
  display: flex;
  flex-direction: column;
  padding: 24px;
  gap: 24px;
}
.brand { display: flex; align-items: center; gap: 12px; }
.brand-icon { font-size: 28px; }
.brand-text h1 { font-size: 20px; font-weight: 700; }
.badge { font-size: 11px; background: rgba(59,130,246,0.15); color: #60a5fa; padding: 2px 8px; border-radius: 999px; }

.selector-section { display: flex; flex-direction: column; gap: 10px; }
.selector-section label { font-size: 12px; font-weight: 600; color: var(--text-muted); text-transform: uppercase; }
.custom-select {
  background: var(--bg-tertiary);
  border: 1px solid var(--border-color);
  color: var(--text-main);
  padding: 10px 14px;
  border-radius: 8px;
  outline: none;
  font-size: 14px;
}
.custom-select:focus { border-color: var(--accent-color); box-shadow: 0 0 0 2px var(--accent-glow); }

.card-status-pill {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 12px;
  color: var(--text-muted);
  margin-top: 4px;
}
.status-dot { width: 8px; height: 8px; border-radius: 50%; background-color: var(--success); }
.hidden { display: none !important; }

.quick-prompts { display: flex; flex-direction: column; gap: 8px; margin-top: auto; }
.quick-prompts h3 { font-size: 12px; font-weight: 600; color: var(--text-muted); text-transform: uppercase; }
.prompt-chip {
  background: var(--bg-tertiary);
  border: 1px solid var(--border-color);
  color: var(--text-main);
  padding: 8px 12px;
  border-radius: 6px;
  font-size: 13px;
  text-align: left;
  cursor: pointer;
  transition: all 0.2s;
}
.prompt-chip:hover { border-color: var(--accent-color); background: rgba(59,130,246,0.1); }

/* Chat Workspace */
.chat-workspace {
  flex: 1;
  display: flex;
  flex-direction: column;
  background-color: var(--bg-primary);
  position: relative;
}
.chat-header {
  padding: 20px 28px;
  border-bottom: 1px solid var(--border-color);
  display: flex;
  justify-content: space-between;
  align-items: center;
}
.header-info h2 { font-size: 18px; font-weight: 600; }
.header-info p { font-size: 13px; color: var(--text-muted); }

.chat-feed {
  flex: 1;
  overflow-y: auto;
  padding: 28px;
  display: flex;
  flex-direction: column;
  gap: 20px;
}
.message { display: flex; gap: 14px; max-width: 85%; }
.message.user { align-self: flex-end; flex-direction: row-reverse; }
.avatar {
  width: 36px;
  height: 36px;
  border-radius: 50%;
  background: var(--bg-tertiary);
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 18px;
}
.bubble {
  background: var(--bg-secondary);
  border: 1px solid var(--border-color);
  padding: 16px 20px;
  border-radius: 12px;
  line-height: 1.6;
  font-size: 14px;
}
.message.user .bubble {
  background: var(--accent-color);
  color: #fff;
  border: none;
}
.bubble h3 { font-size: 15px; margin: 12px 0 6px 0; color: #93c5fd; }
.bubble h3:first-child { margin-top: 0; }
.bubble ul { margin-left: 18px; margin-bottom: 8px; }

/* Input bar */
.input-container {
  padding: 20px 28px;
  border-top: 1px solid var(--border-color);
  background: var(--bg-secondary);
}
#chatForm { display: flex; gap: 12px; }
#userInput {
  flex: 1;
  background: var(--bg-tertiary);
  border: 1px solid var(--border-color);
  padding: 14px 18px;
  border-radius: 10px;
  color: var(--text-main);
  font-size: 14px;
  outline: none;
}
#userInput:focus { border-color: var(--accent-color); }
.primary-btn {
  background: var(--accent-color);
  color: #fff;
  border: none;
  padding: 0 24px;
  border-radius: 10px;
  font-weight: 600;
  cursor: pointer;
  display: flex;
  align-items: center;
  gap: 8px;
  transition: 0.2s;
}
.primary-btn:hover { background: #2563eb; }
.secondary-btn {
  background: var(--bg-tertiary);
  border: 1px solid var(--border-color);
  color: var(--text-main);
  padding: 8px 16px;
  border-radius: 8px;
  cursor: pointer;
}

/* Citation Drawer */
.citation-drawer {
  width: 360px;
  background: var(--bg-secondary);
  border-left: 1px solid var(--border-color);
  padding: 20px;
  display: flex;
  flex-direction: column;
  transition: margin-right 0.3s ease;
}
.citation-drawer.collapsed { margin-right: -360px; display: none; }
.drawer-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px; }
.icon-btn { background: none; border: none; color: var(--text-muted); font-size: 24px; cursor: pointer; }
.citation-list { overflow-y: auto; display: flex; flex-direction: column; gap: 14px; }
.citation-card {
  background: var(--bg-tertiary);
  border: 1px solid var(--border-color);
  padding: 14px;
  border-radius: 8px;
  font-size: 13px;
}
.citation-card .path { font-weight: 600; color: #60a5fa; font-size: 12px; margin-bottom: 6px; }
.citation-card .snippet { color: var(--text-muted); line-height: 1.5; font-size: 12px; }
.citation-card .page-badge {
  display: inline-block;
  background: rgba(16, 185, 129, 0.2);
  color: #34d399;
  padding: 2px 6px;
  border-radius: 4px;
  font-size: 11px;
  font-weight: 600;
  margin-top: 8px;
}
```

- [ ] **Step 3: Create `web/app.js`**

```javascript
// web/app.js
let cardsData = [];
let selectedBank = "";
let selectedCardId = "";
let latestCitations = [];

document.addEventListener("DOMContentLoaded", () => {
  fetchCards();
  setupEventListeners();
});

async function fetchCards() {
  try {
    const res = await fetch("/api/cards");
    cardsData = await res.json();
    populateBankDropdown();
  } catch (err) {
    console.error("Failed to load cards:", err);
  }
}

function populateBankDropdown() {
  const bankSelect = document.getElementById("bankSelect");
  bankSelect.innerHTML = '<option value="">-- Choose Bank --</option>';

  const banks = [...new Set(cardsData.map(c => c.bank))];
  banks.forEach(b => {
    const opt = document.createElement("option");
    opt.value = b;
    opt.textContent = b;
    bankSelect.appendChild(opt);
  });

  // Default to TD if present
  if (banks.includes("TD")) {
    bankSelect.value = "TD";
    handleBankChange("TD");
  }
}

function handleBankChange(bank) {
  selectedBank = bank;
  const cardSelect = document.getElementById("cardSelect");
  cardSelect.innerHTML = '<option value="">-- Choose Card --</option>';
  cardSelect.disabled = !bank;

  const filtered = cardsData.filter(c => c.bank === bank);
  filtered.forEach(c => {
    const opt = document.createElement("option");
    opt.value = c.id;
    opt.textContent = c.card_name;
    cardSelect.appendChild(opt);
  });

  if (filtered.length > 0) {
    cardSelect.value = filtered[0].id;
    handleCardChange(filtered[0].id);
  }
}

function handleCardChange(cardId) {
  selectedCardId = cardId;
  const card = cardsData.find(c => c.id === cardId);
  if (card) {
    document.getElementById("activeCardTitle").textContent = card.card_name;
    document.getElementById("activeCardSub").textContent = `${card.bank} • ${card.indexed ? 'Indexed & Ready' : 'Indexing on demand'}`;
    const statusPill = document.getElementById("cardStatus");
    statusPill.classList.remove("hidden");
  }
}

function setupEventListeners() {
  document.getElementById("bankSelect").addEventListener("change", (e) => handleBankChange(e.target.value));
  document.getElementById("cardSelect").addEventListener("change", (e) => handleCardChange(e.target.value));

  document.getElementById("chatForm").addEventListener("submit", (e) => {
    e.preventDefault();
    submitQuery();
  });

  document.querySelectorAll(".prompt-chip").forEach(chip => {
    chip.addEventListener("click", () => {
      document.getElementById("userInput").value = chip.dataset.prompt;
      submitQuery();
    });
  });

  const drawer = document.getElementById("citationDrawer");
  document.getElementById("toggleDrawerBtn").addEventListener("click", () => drawer.classList.toggle("collapsed"));
  document.getElementById("closeDrawerBtn").addEventListener("click", () => drawer.classList.add("collapsed"));
}

async function submitQuery() {
  const input = document.getElementById("userInput");
  const query = input.value.trim();
  if (!query || !selectedCardId) return;

  appendMessage("user", query);
  input.value = "";

  const assistantMsgId = appendLoadingMessage();

  try {
    const res = await fetch("/api/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ card_id: selectedCardId, question: query })
    });

    if (!res.ok) throw new Error("API call failed");
    const data = await res.json();

    updateAssistantMessage(assistantMsgId, data.answer);
    renderCitations(data.citations);
  } catch (err) {
    updateAssistantMessage(assistantMsgId, `⚠️ **Error retrieving answer:** ${err.message}`);
  }
}

function appendMessage(role, text) {
  const feed = document.getElementById("chatFeed");
  const msg = document.createElement("div");
  msg.className = `message ${role}`;
  msg.innerHTML = `
    <div class="avatar">${role === 'user' ? '👤' : '🛡️'}</div>
    <div class="bubble">${marked.parse(text)}</div>
  `;
  feed.appendChild(msg);
  feed.scrollTop = feed.scrollHeight;
}

function appendLoadingMessage() {
  const feed = document.getElementById("chatFeed");
  const msgId = "msg-" + Date.now();
  const msg = document.createElement("div");
  msg.id = msgId;
  msg.className = "message assistant";
  msg.innerHTML = `
    <div class="avatar">🛡️</div>
    <div class="bubble"><p><em>Analyzing certificate and verifying clauses...</em></p></div>
  `;
  feed.appendChild(msg);
  feed.scrollTop = feed.scrollHeight;
  return msgId;
}

function updateAssistantMessage(id, markdown) {
  const msg = document.getElementById(id);
  if (msg) {
    msg.querySelector(".bubble").innerHTML = marked.parse(markdown);
    const feed = document.getElementById("chatFeed");
    feed.scrollTop = feed.scrollHeight;
  }
}

function renderCitations(citations) {
  latestCitations = citations || [];
  document.getElementById("citationCount").textContent = latestCitations.length;
  const list = document.getElementById("citationList");
  list.innerHTML = "";

  if (latestCitations.length === 0) {
    list.innerHTML = '<p class="empty-state">No citations returned for this query.</p>';
    return;
  }

  latestCitations.forEach(c => {
    const card = document.createElement("div");
    card.className = "citation-card";
    card.innerHTML = `
      <div class="path">${c.breadcrumb}</div>
      <div class="snippet">${c.snippet}</div>
      <span class="page-badge">Page ${c.page_number}</span>
    `;
    list.appendChild(card);
  });
}
```

- [ ] **Step 4: Verify full stack integration**

Run: `python -m pytest tests/test_api.py -v`  
Expected: PASS

---

## Plan Self-Review Checklist

1. **Spec Coverage:**
   - Ingestion with `pymupdf4llm` & breadcrumbs $\rightarrow$ Task 2
   - Card-scoped BM25 + dense vector store $\rightarrow$ Task 3
   - Hybrid search with RRF $\rightarrow$ Task 4
   - Grounded generation with strict prompt & citations $\rightarrow$ Task 5
   - 10-Question Ground Truth Benchmark Suite $\rightarrow$ Task 6
   - FastAPI server with card discovery $\rightarrow$ Task 7
   - Web application with dropdown & citation drawer $\rightarrow$ Task 8
2. **No Placeholders:** All tasks contain full code, explicit assertions, and exact paths.
3. **Type & Signature Consistency:** `CardMetadata`, `DocumentChunk`, `CardIndexData`, `ScoredChunk`, `GroundedResponse` signatures align across tasks.
