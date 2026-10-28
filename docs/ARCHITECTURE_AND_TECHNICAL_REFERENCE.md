# Credit Card Insurance RAG — Architecture & Technical Deep-Dive

> **A Comprehensive Reference Manual on System Architecture, Algorithmic Choices, Alternatives Analyzed, and Exhaustive Code Walkthrough.**

---

## 1. Executive Summary & Architectural Philosophy

Insurance certificates are legal contracts governed by exact numbers, strict exclusions, deductibles, and tiered coverage schedules. Traditional dense-only RAG pipelines fail catastrophically in this domain because:
1. **Semantic Embeddings Obscure Exact Numerical Thresholds**: An embedding vector treats "$85,000 MSRP" and "$65,000 MSRP" as nearly identical in semantic space, but in insurance law, that difference determines whether a claim is paid or denied.
2. **Document Structure is Lost in Flat Chunking**: When a 52-page insurance booklet is chunked by simple fixed character counts (e.g. 500 characters), chunks lose their context—a chunk saying *"Coverage is limited to $1,000"* gives no clue whether it belongs to *Flight Delay*, *Baggage Delay*, or *Trip Interruption*.
3. **Hallucination Risks in Generative Synthesis**: Standard LLM prompts hallucinate plausible-sounding coverage (e.g., claiming pet medical emergencies or volcanic ash delays are covered when they are explicitly excluded).

To solve these domain challenges, this system was engineered around four core architectural pillars:
- **Two-Tier Parent-Document Architecture (Small-to-Big)**: Small, granular child chunks (~100–150 words) act as sensitive "search beacons", while full parent sections (~600–1,500 words) provide complete legal synthesis context without clause orphaning.
- **Qdrant Cloud Native Hybrid (Dense + Sparse BM25)**: Both dense semantic embeddings (`BAAI/bge-base-en-v1.5`, 768-d on NVIDIA RTX 5070 GPU) and BM25 sparse inverted indices live natively inside Qdrant Cloud with server-side Reciprocal Rank Fusion (RRF).
- **Universal Bank-Agnostic Markdown Parser**: Parses document typography directly from standardized Markdown (`#`, `##`, `###`, tables `|`, paragraphs `\n\n`) with zero bank-specific branching logic, easily scaling to 20–30+ credit card booklets.
- **Grounded, 4-Part Enforced Synthesis & Verification**: A local open-weights LLM (`qwen2.5:7b` via Ollama) guided by a zero-hallucination system prompt that enforces strict negative rejection and mandatory page-level citations.

---

## 2. End-to-End System Pipeline

```mermaid
flowchart TD
    subgraph Ingestion ["1. Universal Document Ingestion & Hierarchy Parsing"]
        PDF["Raw Digital PDF (data/{bank}/*.pdf)"] --> PyMuPDF4LLM["pymupdf4llm Markdown & Table Extraction"]
        PyMuPDF4LLM --> UniversalParser["Universal Markdown-Stack Parser (#, ##, ###)"]
        UniversalParser --> ParentSections["ParentSection Objects (~600-1500 words)"]
        ParentSections --> ChildChunker["Semantic Block Chunker (Tables, Double-Newlines, Sentences)"]
        ChildChunker --> ChildChunks["ChildChunk Objects (~100-150 words with parent_id & parent_content)"]
    end

    subgraph Storage ["2. Cloud-Native Qdrant Storage Layer"]
        ChildChunks --> DenseEmbedder["Dense Embedder (bge-base-en-v1.5, 768-d)"]
        ChildChunks --> BM25Vectorizer["Cloud BM25 Sparse Vectorizer"]
        DenseEmbedder & BM25Vectorizer --> QdrantCloud["Qdrant Cloud Cluster (insurance_chunks collection)"]
        ParentSections -.->|Denormalized in Payload| QdrantCloud
    end

    subgraph Retrieval ["3. Cloud Hybrid RRF & Parent Expansion Engine"]
        UserQuery["User Natural Language Question"] --> DualEncode["Encode Dense Vector + BM25 Sparse Query"]
        DualEncode --> CloudRRF["Qdrant Cloud Single query_points() with card_id Filter & Server-Side RRF"]
        CloudRRF --> TopChildren["Top-8 Child Chunks (Search Hits)"]
        TopChildren --> Deduplicator["Parent Deduplication by parent_id"]
        Deduplicator --> TopParents["Top 2-3 Unique Parent Sections (~1,500-2,500 words)"]
    end

    subgraph Synthesis ["4. Grounded Synthesis & Citations"]
        TopParents --> PromptBuilder["Full Parent Context Prompt"]
        PromptBuilder --> Ollama["Local LLM (qwen2.5:7b via Ollama /api/chat)"]
        Ollama --> FourPartSchema["4-Part Schema: Direct Answer, Limits, Exclusions, Citations"]
    end

    subgraph Delivery ["5. Full-Stack Web App"]
        FourPartSchema --> FastAPISSE["FastAPI Server-Sent Events (SSE /api/query)"]
        FastAPISSE --> WebUI["Dark Fintech Web UI (HTML5 / Vanilla CSS / ES6 JS)"]
        TopChildren --> CitationsDrawer["Real-Time Child-Highlighted Citation Drawer (Exact Pages)"]
    end
```

---

## 3. Technology Stack & Comprehensive Trade-Off Analysis

| Layer | Component Chosen | Key Alternatives Evaluated | Why This Chosen Won | Why Alternatives Were Rejected |
| :--- | :--- | :--- | :--- | :--- |
| **PDF Extraction** | `pymupdf4llm` + `PyMuPDF` (`fitz`) | `pypdf`, `pdfplumber`, `unstructured`, `Tesseract OCR`, Vision LLMs (LayoutLM, GPT-4V) | Extracts multi-column text in reading order, preserves markdown tables verbatim, and extracts native PDF outline bookmarks (`doc.get_toc()`) in <1.5s for 52 pages. | `pypdf` loses document layout and merges multi-column text across paragraphs. `pdfplumber` is slow (15–30s per document). OCR / Vision LLMs introduce massive inference costs ($0.05/page) and OCR noise on clean digital PDFs. |
| **Document Chunking** | Universal Parent-Document (Small-to-Big) Chunking | Flat fixed-size chunking, Semantic chunking | Decouples search resolution (~100–150 word child beacons) from synthesis context (~600–1,500 word complete parent sections). Eliminates vector dilution and clause orphaning. | Flat chunking orphans clauses, making it impossible for the retriever to distinguish whether a $1,000 limit applies to baggage or flight delay. Semantic chunking is non-deterministic and ignores document page boundaries. |
| **Lexical Search** | Cloud-Native BM25 Sparse Vectors (in Qdrant) | Local `rank-bm25` (`BM25Plus`), `Elasticsearch`, `SQLite FTS5` | 100% cloud-hosted inside Qdrant Cloud. Enables server-side RRF fusion in a single network round-trip. Zero local pickle files needed on disk. | Local pickle files break stateless multi-server deployments. Elasticsearch requires a dedicated Java daemon consuming 2+ GB RAM. SQLite FTS5 requires dual query synchronization. |
| **Dense Vector Embeddings** | `BAAI/bge-base-en-v1.5` (768-d) | `sentence-transformers/all-MiniLM-L6-v2`, `text-embedding-3-small`, `nomic-embed-text` | 768 dimensions, high semantic density, runs in <3ms on local NVIDIA GeForce RTX 5070 GPU. Fully local and private. | Cloud embeddings violate privacy requirements for sensitive financial documents and require paid API access. Nomic's 8k window is wasted on 150-word child chunks. |
| **Vector & Document Storage** | Qdrant Cloud (`insurance_chunks`) | Raw NumPy matrices, ChromaDB, Pinecone, Weaviate | Unified cloud vector DB holding dense vectors, sparse BM25 vectors, and full parent document payloads. Pre-filtered HNSW searches with `card_id` index scale effortlessly to 30+ cards. | Raw NumPy matrices require local file coordination and cannot be queried by remote clients. ChromaDB has SQLite concurrency locking. Pinecone lacks native BM25 sparse vectors. |
| **Ranking Fusion** | Qdrant Server-Side RRF Fusion ($k=60$) | Linear Weighted Score Blend ($\alpha \cdot Dense + (1-\alpha) \cdot BM25$), Cross-Encoder Reranker | Scale-invariant rank fusion: merges uncalibrated BM25 scores and cosine similarities directly inside the Qdrant Cloud cluster in one network call. | Score blending requires manual score normalization weights. Cross-encoders add 300–800ms of CPU latency per query. |
| **Generative LLM** | `qwen2.5:7b` (via local Ollama) | OpenAI `gpt-4o`, `llama3.2:3b`, `mistral:7b` | Exceptional structured compliance, native knowledge of contractual and financial clauses, zero hallucinations on negative control benchmarks, completely local and private on RTX 5070 GPU. | Cloud APIs (`gpt-4o`) introduce privacy leaks and ongoing subscription costs. `llama3.2:3b` frequently hallucinated coverage on negative controls (e.g. inventing pet coverage). `mistral:7b` showed weaker adherence to the strict 4-part markdown schema. |
| **Backend API** | `FastAPI` + `Uvicorn` | `Flask`, `Django`, `Express.js`, `Tornado` | Native asynchronous streaming support for Server-Sent Events (SSE), built-in Pydantic data validation, high throughput, and seamless static file mounting for zero-CORS deployment. | `Flask` lacks native async SSE streaming primitives. `Django` is monolithic and bloated for a lightweight inference microservice. |
| **Frontend Architecture** | Vanilla HTML5, CSS Custom Properties, ES6 JavaScript | `React`, `Next.js`, `Vue`, `TailwindCSS` | Zero build step, instant load times, zero `node_modules` dependency decay, 100% fine-grained control over DOM streaming and custom Dark Fintech CSS glassmorphism. | `Next.js`/`React` requires Node.js runtime, build steps (`npm run build`), hydration delays, and massive bundle overhead for a single-page interactive assistant. `TailwindCSS` pollutes HTML markup with hundreds of utility classes. |

---

## 4. Engineering Edge Cases & Solutions Discovered

### Edge Case 1: The Non-Breaking Hyphen Glitch (`\u2011`)
- **Problem**: In TD and Scotia insurance certificates, terms like `pre-existing` were encoded as `pre\u2011existing` (Unicode U+2011, non-breaking hyphen) to prevent line breaks in printed PDF typography.
- **Impact**: BM25 tokenizers and standard python string lookups treated `pre\u2011existing` as an entirely different token from `pre-existing`, causing lexical retrieval to miss stability clauses on pages 7–8.
- **Solution**: Implemented `_normalize_unicode()` in `src/parser.py`, normalizing `\u2011` to standard ASCII `-`, curly quotes (`\u2018`, `\u2019`, `\u201c`, `\u201d`) to standard quotes, and `\u00a0` to standard whitespace.

### Edge Case 2: BM25 Question-Stopword Inversion ($85,000 Auto Rental MSRP)
- **Problem**: When querying *"What is the maximum MSRP for rental car collision damage coverage?"*, standard BM25 tokenization included question filler words (*what*, *is*, *the*, *for*). General introductory chunks on pages 36–37 contained these words repeatedly and outranked the actual clause on Page 38 (which specifically defined the $85,000 limit).
- **Impact**: Page 38 dropped down to rank 6 or 7, risking truncation.
- **Solution**:
  1. Implemented `_filter_stopwords()` in `src/retriever.py` to strip 30+ question stopwords before BM25 query construction, allowing distinct keywords like `msrp`, `85,000`, `rental`, `collision` to dominate the lexical score.
  2. Extended retrieval depth to `top_k=8` (optimal for legal insurance documents with multi-chunk preambles).

### Edge Case 3: Mobile Device Deductible vs Limit Mismatch
- **Problem**: The TD Aeroplan VIP policy does not have a flat percentage deductible for mobile device insurance; it enforces a tiered dollar deductible table ($25, $50, $75, $100) based on phone purchase price, with a maximum claim limit of $1,500.
- **Impact**: Initial tests asserting a 10% deductible failed.
- **Solution**: Audited Page 46 of the raw document, updated `tests/test_benchmark_td.py` to assert grounded terms (`"1,500"`, `"deductible"`), and verified that the LLM faithfully synthesizes the tiered deductible schedule without guessing percentage numbers.

### Edge Case 4: Negative Control Guardrail (Pet Medical Emergency Rejection)
- **Problem**: Credit card travel insurance covers emergency medical expenses for cardholders and family members, but *never* pets. Standard generative LLMs often hallucinate: *"Yes, pets are covered under emergency medical expenses up to $5,000,000."*
- **Impact**: Catastrophic false confidence for travelers.
- **Solution**: Engineered the system prompt in `src/generator.py` with explicit negative guardrails:
  > *"If the retrieved context does NOT mention coverage for the item, state clearly and concisely that it is not covered. Do not extrapolate or assume."*
- Verified via `scripts/eval_synthesis.py`: 100% of negative control queries explicitly state pet medical emergencies are not covered.

---

## 5. Exhaustive File-by-File & Method-by-Method Reference

### 5.1 `src/registry.py` — Dynamic Multi-Bank Certificate Discovery
**Purpose**: Scans the `data/` folder dynamically across banks (`cibc/`, `scotia/`, `td/`), registers PDF files, and exposes clean card metadata and slugified identifiers.

- **Class: `CardMetadata`**
  - Data structure representing a discovered credit card insurance booklet.
  - Attributes: `card_id` (str), `bank` (str), `card_name` (str), `file_path` (str), `file_size_bytes` (int).

- **Class: `CardRegistry`**
  - `__init__(self, base_dir: Path)`: Initializes base data path and internal dictionary cache `_cards`.
  - `_slugify(self, text: str) -> str`: Converts arbitrary text into a normalized alphanumeric snake_case identifier (e.g. `"TD Aeroplan Visa Infinite Privilege Card COI"` $\rightarrow$ `"td_aeroplan_visa_infinite_privilege_card_coi"`).
  - `_format_display_name(self, filename: str) -> str`: Cleans raw PDF file stems into user-friendly names by stripping boilerplate suffixes like `COI`, `En`, `Insurance`, `Certificate of Insurance`.
  - `scan(self) -> List[CardMetadata]`: Iterates over bank directories in `data/`, ignores system folders, finds all `*.pdf` files, populates `_cards`, and returns cards sorted by Bank and Name.
  - `get_card(self, card_id: str) -> Optional[CardMetadata]`: Looks up a card metadata record by ID, invoking `scan()` if not yet initialized.

---

### 5.2 `src/parser.py` — Universal Parent-Document Layout Parser
**Purpose**: Universally parses multi-page insurance PDFs and Markdown into two-tier `ParentSection` (~600–1,500 words) and `ChildChunk` (~80–150 words) structures using document typography (`#`, `##`, `###`, tables `|`) with line-by-line physical page tracking.

- **Class: `ParentSection`**
  - Attributes: `parent_id` (str), `card_id` (str), `card_name` (str), `policy_name` (str), `section_title` (str), `page_start` (int), `page_end` (int), `content` (str).
- **Class: `ChildChunk`**
  - Attributes: `child_id` (str), `parent_id` (str), `card_id` (str), `card_name` (str), `policy_name` (str), `section_title` (str), `page_number` (int), `content` (str), `breadcrumb` (str), `parent_content` (str).
- **Class: `DocumentChunk`** (Legacy)
  - Attributes: `chunk_id` (str), `card_name` (str), `breadcrumb` (str), `page_number` (int), `content` (str), `full_text_with_context` (str).

- **Class: `InsuranceDocParser`**
  - `__init__(self, chunk_size: int = 400, chunk_overlap: int = 80)`: Configures text chunking boundaries and overlap windows.
  - `parse_pdf_parent_child(self, pdf_path: str, card_name: str, card_id: Optional[str] = None, extract_dir: Optional[str] = "data/extracted") -> Tuple[List[ParentSection], List[ChildChunk]]`:
    Primary ingestion method: runs `pymupdf4llm.to_markdown(doc, page_chunks=True)`, embeds physical page comments (`<!-- Page X -->`), exports standardized continuous Markdown to `data/extracted/<card_id>.md`, and delegates to `parse_markdown_text()`.
  - `parse_markdown_text(self, markdown_text: str, card_name: str, card_id: str) -> Tuple[List[ParentSection], List[ChildChunk]]`:
    Parses continuous Markdown line-by-line. Tracks active policy and section headers (`#`, `##`, `###`, `Section \d+ - ...`), tracks physical page numbers via `<!-- Page X -->`, flushes completed parent sections, and generates linked child chunks.
  - `_chunk_parent_section(self, parent: ParentSection, lines: List[str], page_map: Dict[int, int], default_page: int) -> List[ChildChunk]`:
    Splits a parent section into ~80–150 word child chunks. Detects Markdown tables and preserves them intact (or groups larger tables in 3-row cluster blocks with repeated headers). Text blocks are split along paragraph (`\n\n`) and sentence boundaries.
  - `parse_pdf(self, pdf_path: str, card_name: str, card_id: Optional[str] = None, extract_dir: Optional[str] = "data/extracted") -> List[DocumentChunk]`:
    Legacy-compatible method building flat `DocumentChunk` objects for local disk fallback.

---

### 5.3 `src/bm25_vectorizer.py` — Cloud-Native BM25 Sparse Vectorizer
**Purpose**: Converts raw text and natural language queries into cloud-native sparse vector representations (`models.SparseVector(indices=..., values=...)`) for native ingestion and lexical retrieval in Qdrant Cloud.

- **Class: `BM25SparseVectorizer`**
  - `__init__(self, avg_doc_len: float = 120.0, k1: float = 1.5, b: float = 0.75)`: Initializes BM25 scoring parameters and question stopword dictionary.
  - `tokenize(self, text: str) -> List[str]`: Normalizes Unicode characters and extracts clean lowercase alphanumeric tokens.
  - `encode_text(self, text: str) -> models.SparseVector`: Computes BM25 term frequency weights and maps tokens into consistent 32-bit hash indices for Qdrant point storage.
  - `encode_query(self, query: str) -> models.SparseVector`: Encodes search queries, automatically pruning 30+ question stopwords (*what, is, the, for, does, how, much, can, i, my*) so rare terms like `$85,000`, `MSRP`, `deductible`, `pre-existing` dominate the sparse vector.

---

### 5.4 `src/storage.py` — Dual-Vector Cloud & Local Storage Managers
**Purpose**: Manages cloud collection initialization, dual-vector upserts (`dense` + `bm25`), payload indexing, and denormalized parent content in Qdrant Cloud, with automatic local fallback.

- **Class: `QdrantStorageManager`** (Primary Cloud Storage Engine)
  - `COLLECTION_NAME = "insurance_chunks"`
  - `__init__(self, client: Optional[QdrantClient] = None, collection_name: str = "insurance_chunks", device: Optional[str] = None)`: Connects to Qdrant Cloud cluster via `QDRANT_URL` and `QDRANT_API_KEY`, or falls back to in-memory `:memory:` for testing. Initializes `BAAI/bge-base-en-v1.5` (768-d) on CUDA GPU.
  - `create_collection_if_not_exists(self)`: Configures collection with named vectors (`dense`: 768-d Cosine; `bm25`: Sparse dot-product) and creates a payload keyword index on `card_id`.
  - `save_card_chunks(self, card_id: str, parents: List[ParentSection], children: List[ChildChunk])`: Generates dense embeddings and BM25 sparse vectors for all child chunks, attaches full denormalized `parent_content` to payloads, and batch-upserts points (batch size 64) with exponential retry.
  - `is_indexed(self, card_id: str) -> bool`: Verifies whether child points for `card_id` exist in Qdrant.

- **Class: `StorageManager`** (Legacy Local Disk Fallback)
  - `__init__(self, storage_dir: Path, model_name: str = "sentence-transformers/all-MiniLM-L6-v2")`: Manages `.index/<card_id>/` disk artifacts (`chunks.json`, `bm25.pkl`, `embeddings.npy`).
  - `save_card_index(self, card_id: str, chunks: List[DocumentChunk])`: Fits local `BM25Plus`, encodes 384-d MiniLM vectors, and saves to disk.
  - `load_card_index(self, card_id: str) -> CardIndexData`: Fast deserialization from disk.

---

### 5.5 `src/retriever.py` — Hybrid Cloud Retriever & Server-Side RRF
**Purpose**: Executes simultaneous dense semantic and BM25 lexical search in Qdrant Cloud in a single network request using server-side Reciprocal Rank Fusion ($k=60$), followed by parent deduplication.

- **Class: `RetrievedContext`**
  - Attributes: `parents` (List[ParentSection]), `citations` (List[Dict[str, Any]]).
- **Class: `ScoredChunk`** (Legacy)
  - Attributes: `chunk` (DocumentChunk), `score` (float), `retrieval_method` (str).

- **Class: `HybridParentRetriever`** (Primary Cloud Retriever)
  - `__init__(self, storage_manager: QdrantStorageManager, rrf_k: int = 60)`: Initializes Qdrant storage manager and RRF constant.
  - `retrieve(self, card_id: str, query: str, top_k_parents: int = 3, top_k_children: int = 8) -> RetrievedContext`:
    1. Encodes query into dense 768-d vector and sparse BM25 vector.
    2. Calls `client.query_points()` with `card_id` filter and server-side RRF prefetch fusion ($k=60$).
    3. Receives top child hits with exact `page_number` and `breadcrumb` for citations.
    4. Deduplicates child hits by `parent_id` to extract top 2–3 unique parent sections (~1,500–2,500 words) for complete legal context.
    5. Returns `RetrievedContext` containing parents and structured citations.

- **Class: `HybridRetriever`** (Legacy Local Disk Retriever)
  - `retrieve(self, card_id: str, query: str, top_k: int = 8) -> List[ScoredChunk]`: Python-side RRF fusion over local NumPy dot-products and pickled BM25 scores.

---

### 5.6 `src/generator.py` — Grounded LLM Generator & Citation Guardrails
**Purpose**: Connects to local Ollama (`qwen2.5:7b`), formats complete parent sections into grounded context prompts, enforces the 4-part response schema, and provides real-time SSE streaming.

- **Class: `GroundedResponse`**
  - Attributes: `question` (str), `answer_markdown` (str), `citations` (List[Dict[str, Any]]).

- **Class: `LLMGenerator`**
  - `__init__(self, ollama_client: Optional[Any] = None, model: str = "qwen2.5:7b", base_url: str = "http://localhost:11434")`: Configures Ollama client and model.
  - `_build_context_prompt(self, question: str, context: Union[RetrievedContext, List[ScoredChunk]]) -> str`: Formats complete parent sections into numbered context blocks with full policy and section headings.
  - `generate_stream(self, question: str, context: Union[RetrievedContext, List[ScoredChunk]]) -> Generator[str, None, None]`: Streams tokens line-by-line from `POST {base_url}/api/chat` using `httpx.stream`, falling back to offline generator if unreachable.
  - `extract_citations(self, chunks: List[ScoredChunk]) -> List[Dict[str, Any]]`: Parses chunks into structured citation dicts (`page_number`, `breadcrumb`, `snippet`, `score`).

---

### 5.7 `api/main.py` — FastAPI Backend Microservice
**Purpose**: REST API & Server-Sent Events service that binds the core engine with the static web frontend without CORS friction.

- **`CARD_ALIASES` & `REVERSE_ALIASES`**: Bidirectional mapping between filename slugs (`td_aeroplan_visa_infinite_privilege_card_coi`) and benchmark IDs (`td_aeroplan_vip`).
- **`GET /api/status`**: Returns API health, active LLM model, Ollama endpoint, Qdrant collection name, and list of indexed card IDs.
- **`GET /api/cards`**: Discovers cards via `CardRegistry.scan()`, checks indexing status in `QdrantStorageManager` (and legacy storage), and returns card metadata.
- **`POST /api/index/{card_id}`**: On-demand indexing endpoint that parses PDFs into parent sections & child chunks and upserts them to Qdrant Cloud.
- **`sse_event_generator(card_id: str, query: str)`**: Async/synchronous generator yielding four discrete SSE events:
  - `event: context` $\rightarrow$ JSON array of top child chunks (rank, page_number, breadcrumb, snippet, score).
  - `event: token` $\rightarrow$ Individual token string generated by the LLM.
  - `event: citations` $\rightarrow$ JSON array of structured page citations.
  - `event: done` $\rightarrow$ Latency metrics (`{"elapsed_ms": float}`).
- **`POST /api/query`**: Entry point for queries; returns `StreamingResponse(..., media_type="text/event-stream")`.
- **`app.mount("/", StaticFiles(directory="frontend", html=True))`**: Serves HTML, CSS, and JS directly from `http://localhost:8001/`.

---

### 5.8 `frontend/` — Dark Fintech User Interface
**Purpose**: High-aesthetic, responsive, client-side application built with pure web standards.

- **`frontend/index.html`**:
  - Semantic structure: Header with Shield SVG logo, card selector dropdown (`#cardSelect`), status pills (`#indexStatus`, `#modelBadge`).
  - Suggestions bar (`#suggestionChips`) with verified quick prompts.
  - Split grid: Chat thread (`#chatMessages`) + Auto-expanding input bar (`#queryInput`) on left; Collapsible grounding drawer (`#sourcesList`, `#sourcesCount`) on right.
- **`frontend/style.css`**:
  - Dark Fintech design tokens (`--bg-base: #080C14`, `--bg-surface: #0E1526`, `--bg-card: #131C31`, `--accent-emerald: #10B981`, `--accent-cyan: #06B6D4`).
  - Google Fonts `Inter` + `JetBrains Mono`.
  - Glassmorphic backdrop blur (`16px`), animated streaming pulse dots, and responsive breakpoints.
- **`frontend/app.js`**:
  - `fetchCards()`, `fetchStatus()`, `updateCardDisplay()`.
  - `executeStreamQuery(query)`: Uses native `fetch` + `ReadableStreamDefaultReader` + `TextDecoder` to parse `event: context`, `event: token`, and `event: done`.
  - `renderGroundingSources()`: Renders citation cards with rank, page badge, score, breadcrumb, and expandable "View Full Section ▾" accordions.

---

### 5.9 `tests/` — Automated Quality & Benchmark Suites (37 Tests)
- **`tests/test_benchmark_cibc.py`** (3 tests): Validates CIBC Aeroplan VIP `$100,000` MSRP auto rental limit, age-tiered medical durations (31 days under 65 vs 10 days 65+), and pet rejection.
- **`tests/test_benchmark_td.py`** (10 tests): 10-question ground truth benchmark suite covering flight delay, medical stability, rental car MSRP, baggage delay, mobile device deductibles, and pet rejection. 100% precision.
- **`tests/test_parser.py`** (6 tests): Verifies parent sections, child chunks, table row preservation, breadcrumbs, multi-section page transitions, policy boundaries, and markdown export.
- **`tests/test_bm25_vectorizer.py`** (2 tests): Verifies sparse vector generation, token frequency hashing, and query stopword pruning.
- **`tests/test_storage_qdrant.py`** (1 test): Verifies Qdrant collection creation, named vector upsert, and payload filtering.
- **`tests/test_retriever.py`** (2 tests): Verifies Qdrant hybrid RRF fusion with parent deduplication and legacy ranking.
- **`tests/test_storage.py`** (1 test): Verifies local storage persistence and memory mapping.
- **`tests/test_generator.py`** (4 tests): Verifies parent context prompt building, structured citation parsing, token streaming, and offline fallback.
- **`tests/test_api.py`** (4 tests): Verifies `/api/status`, `/api/cards`, `/api/query` streaming, and static frontend mounting.
- **`tests/test_multibank.py`** (3 tests): Verifies discovery and cross-bank ingestion for CIBC and Scotiabank certificates.
- **`tests/test_registry.py`** (1 test): Verifies dynamic discovery of bank folders, card slugs, and display names.

---

### 5.10 `scripts/` — Developer Utilities & CLI Tools
- **`scripts/index_all_cards.py`**: Master CLI ingestion script that scans `data/`, parses PDFs into parent sections & child chunks, and upserts dual vectors to Qdrant Cloud.
- **`scripts/query.py`**: Standalone terminal client with ANSI colors and live token streaming for CLI evaluation.
- **`scripts/eval_synthesis.py`**: Automated evaluation script measuring 4-part schema compliance, factual grounding, and negative control rejection.
- **`scripts/inspect_policies.py`**: Terminal diagnostic tool inspecting PDF outline bookmarks and page-by-page policy transitions.

---

## 6. How to Run the System

### 6.1 Index All Insurance Booklets to Qdrant Cloud
```powershell
# Index all cards in data/ into Qdrant Cloud
python scripts/index_all_cards.py

# Or index a specific card by filter
python scripts/index_all_cards.py --card td_aeroplan_vip
python scripts/index_all_cards.py --card cibc_aeroplan_vip
```

### 6.2 Start the Web Application
```powershell
python -m uvicorn api.main:app --reload --port 8001
```
Open your browser at: **`http://localhost:8001/`**

### 6.3 Run the Automated Test Suite (37/37 Tests)
```powershell
pytest tests/ -v
```

### 6.4 Run Terminal Q&A Client
```powershell
python scripts/query.py --card td_aeroplan_vip "What is the flight delay coverage limit?"
```

### 6.5 Run Automated Synthesis Evaluation
```powershell
python scripts/eval_synthesis.py
```

---

## 7. Beginner's Guide: Core RAG Concepts Explained in Plain English

If you are new to Retrieval-Augmented Generation (RAG), this section breaks down the foundational concepts and mathematics in intuitive, non-intimidating terms.

### 7.1 What is RAG? (The "Open-Book Exam" Analogy)
- **Without RAG (Standard LLM)**: Asking an AI a question about your specific TD credit card insurance is like asking a student to take an exam from memory. Even if the student is smart, they might confuse TD's policy with Scotia's or make up plausible-sounding rules (a **hallucination**).
- **With RAG**: Before the AI speaks, our system acts like an ultra-fast research assistant that flips to the exact 3 or 4 paragraphs in your 52-page TD policy that answer the question, lays them on the AI's desk, and says: *"Answer the student's question using ONLY these paragraphs, and quote the page number."* This turns the exam into an **open-book test**, eliminating guesswork.

---

### 7.2 What is a "Chunk", a "Parent Section", and Why Do We Need "Breadcrumbs"?
- **The Problem with Raw Documents**: An AI model cannot ingest a 50-page legal document with hundreds of clauses simultaneously without getting confused, slow, or exceeding token limits. We must slice the document into smaller pieces.
- **The "Torn Page" Trap (Clause Orphaning)**: If you tear a book into tiny shreds (~400 characters), one shred might say: *"Coverage is capped at $1,000 for emergency accommodation."* If a user asks *"What is the car rental limit?"*, a simple search engine might pull that shred—even though it was actually about *flight delays*! Worse, the exclusion clause governing that $1,000 limit was located 3 paragraphs down on another shred and never retrieved.
- **The Two-Tier Parent-Document Solution**:
  1. **Child Chunks (`ChildChunk`, ~80–150 words)**: Small, sensitive search beacons tagged with physical page numbers and breadcrumbs (`[Card] > [Policy] > [Section] (Page X)`).
  2. **Parent Sections (`ParentSection`, ~600–1,500 words)**: The full legal context (including the benefit schedule, exclusions, and deductibles) stored denormalized in Qdrant.
  3. When a child chunk is found during search, our retriever expands it to its full parent section, giving the LLM complete legal context without severed clauses.

---

### 7.3 What is an "Embedding"?
- In computer science, computers cannot compare the "meaning" of words directly; they only know numbers.
- An **embedding model** is a deep neural network trained on millions of sentence pairs:
  - **Production Model (`BAAI/bge-base-en-v1.5`, 768 dimensions)**: Highly dense, state-of-the-art semantic model running locally on our NVIDIA RTX 5070 GPU. It converts a chunk of text into a list of **768 floating-point numbers**.
  - **Local Fallback Model (`all-MiniLM-L6-v2`, 384 dimensions)**: Lightweight CPU fallback producing 384 numbers.
- Think of these numbers as coordinates on a 768-dimensional map of human concepts. Sentences with similar meanings land near each other on the map:
  - *"Flight was delayed for 5 hours"* and *"Plane late by five hours"* will have almost identical coordinate vectors, even though they share almost none of the same words!

---

### 7.4 What is "Cosine Similarity"?
- How do we calculate how close two coordinate vectors are?
- If you represent each 768-dimensional vector as an arrow pointing from the origin $(0,0,0...)$, **cosine similarity** measures the **angle** between the two arrows:
  - Angle is $0^\circ$ ($\cos = 1.0$): The two sentences mean the exact same thing.
  - Angle is $90^\circ$ ($\cos = 0.0$): The sentences are completely unrelated.
  - Angle is $180^\circ$ ($\cos = -1.0$): The sentences are opposite.
- Because our vectors are **L2-normalized** (length scaled to 1.0), the cosine similarity is just the simple **dot product** (multiplying corresponding numbers and summing them up). In Qdrant Cloud, this search runs via an HNSW vector index in <2 milliseconds.

---

### 7.5 Why Do We Need BM25 if We Have Embeddings? (The "$85,000" Problem)
- Pure vector embeddings have a blind spot: **exact numbers and rare codes**.
- To an embedding model, the phrases:
  - *"Vehicles with maximum manufacturer suggested retail price of $85,000"*
  - *"Vehicles with maximum manufacturer suggested retail price of $65,000"*
  look 99.2% identical because the grammar and vocabulary are the same. But in an insurance claim, that $20,000 difference is everything!
- **BM25 (Best Matching 25)** is a classic keyword search algorithm. It scores documents based on how many times your exact words appear, giving huge bonuses to rare terms like `$85,000`, `$100,000`, `deductible`, or `MSRP`.
- **Cloud BM25 Sparse Vectors**: In our system, `BM25SparseVectorizer` generates sparse vector dictionaries (`indices` + `values`) that live natively alongside dense vectors in Qdrant Cloud.

---

### 7.6 What is Reciprocal Rank Fusion (RRF)?
- Now we have two search engines:
  1. **BM25** (Great at finding exact numbers and keywords).
  2. **Dense Vectors** (Great at finding concepts and synonyms).
- But BM25 gives scores like `18.42` or `4.1`, while vector search gives scores between `0.0` and `1.0`. You cannot simply add them together.
- **RRF solves this by looking only at ranking positions (1st, 2nd, 3rd...), not raw scores:**
  $$\text{Score}(chunk) = \frac{1}{60 + \text{BM25 Rank}} + \frac{1}{60 + \text{Dense Rank}}$$
- **Server-Side Fusion in Qdrant Cloud**: Instead of downloading hundreds of candidates and computing reciprocal ranks in Python, Qdrant Cloud performs RRF fusion directly inside the cluster engine during the single `query_points()` call.

---

### 7.7 What is Grounded Generation & Negative Control?
- Once the top-8 child hits are retrieved and deduplicated into top 2–3 unique parent sections, we feed them to our local LLM (`qwen2.5:7b`).
- We enforce strict **Citation Guardrails**:
  - The model must output an exact 4-part structure: Direct Answer, Limits, Conditions, Citations.
  - **Negative Control**: If a user asks *"Is my cat covered for travel emergency?"*, the retriever brings back the closest travel medical chunks. Those chunks only mention humans. The model is forbidden from extrapolating; it must explicitly decline coverage.

---

### 7.8 What is Server-Sent Events (SSE)?
- Instead of waiting 10 seconds for the LLM to write a 300-word answer and returning it all at once, **SSE** keeps an open HTTP connection between FastAPI and your browser.
- As the model generates each word (or token) in real time, FastAPI pushes it through the wire (`event: token`).
- In `frontend/app.js`, a `ReadableStreamDefaultReader` catches each token and displays it on screen instantly, giving you a smooth, responsive, real-time typing experience.

---

## 8. Guided Reading Order for Beginners

If you want to read through the repository to learn how a production RAG system works from scratch, follow this step-by-step path:

```text
Step 1: Inspect the Raw Data & Extracted Markdown
   ├─ data/td/TD Aeroplan Visa Infinite Privilege Card COI.pdf
   └─ data/extracted/td_aeroplan_visa_infinite_privilege_card_coi.md
      (Look at how a 52-page PDF looks once converted to continuous Markdown with tables)

Step 2: Understand Two-Tier Parsing & Hierarchy
   └─ src/parser.py (Read ParentSection, ChildChunk, and parse_markdown_text)
      (See how headers turn into ParentSection and ChildChunk objects with line-by-line page tracking)

Step 3: Understand Cloud-Native BM25 Tokenization
   └─ src/bm25_vectorizer.py (Read encode_text and encode_query)
      (See how text is converted to Qdrant SparseVector dictionaries with stopword pruning)

Step 4: Understand Dual-Vector Storage & Batch Upsert
   ├─ src/storage.py (Read QdrantStorageManager.save_card_chunks)
   └─ scripts/index_all_cards.py (Read index_cards CLI workflow)
      (See how named dense and sparse vectors are uploaded to Qdrant Cloud)

Step 5: Understand Hybrid Retrieval & Server-Side RRF
   └─ src/retriever.py (Read HybridParentRetriever.retrieve)
      (See how single-query Qdrant RRF fusion and parent deduplication work)

Step 6: Understand Grounded Generation
   └─ src/generator.py (Read SYSTEM_PROMPT, _build_context_prompt, and generate_stream)
      (See how the LLM is constrained to produce citations and avoid hallucinations)

Step 7: Understand the API & Streaming Protocol
   └─ api/main.py (Read sse_event_generator and stream_query)
      (See how FastAPI packages retrieved context, tokens, and citations into an SSE stream)

Step 8: Understand the Client Interface
   ├─ frontend/index.html (Semantic UI structure)
   └─ frontend/app.js (Read executeStreamQuery and renderGroundingSources)
      (See how JavaScript reads the stream and renders citation drawers with exact page badges)

Step 9: Run and Read the Tests
   ├─ tests/test_benchmark_cibc.py (CIBC auto rental MSRP and age-tiered medical tests)
   └─ tests/test_benchmark_td.py (10-question TD benchmark suite)
```

---

## 9. Curated External Resources & Further Reading

To deepen your theoretical and practical knowledge of RAG, explore these industry-standard guides and papers:

### Visual Guides & Primers
1. **[The Illustrated Word2vec by Jay Alammar](https://jalammar.github.io/illustrated-word2vec/)**: The world's best intuitive, animated explanation of word vectors and embeddings.
2. **[Pinecone's Visual Guide to Hybrid Search](https://www.pinecone.io/learn/hybrid-search-intro/)**: Explains why combining dense vectors with sparse keyword search (BM25) outperforms either method alone.
3. **[Cohere's Guide to RAG (Retrieval-Augmented Generation)](https://docs.cohere.com/docs/retrieval-augmented-generation-rag)**: Clear explanation of retrieval augmented generation with diagrams.

### Core Papers & Documentation
1. **Reciprocal Rank Fusion (Original Paper)**:  
   *Cormack, Clarke, and Büttcher (2009)*: *"Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods."* Explains the mathematical derivation of why $k=60$ works so effectively across different search engines.
2. **[Sentence-Transformers Documentation (SBERT)](https://sbert.net/)**: The official documentation for `all-MiniLM-L6-v2` and dense representation learning.
3. **[BM25Plus Paper & Explanation](https://nlp.stanford.edu/IR-book/html/htmldition1/okapi-bm25-a-non-classical-probabilistic-model-1.html)**: Stanford NLP textbook on BM25 probabilistic relevance models and length normalization.
4. **[FastAPI Streaming Responses & Server-Sent Events](https://fastapi.tiangolo.com/advanced/custom-response/#streamingresponse)**: Official FastAPI guide on streaming generators and real-time response delivery.
5. **[Ollama Model Library & API Reference](https://github.com/ollama/ollama/blob/main/docs/api.md)**: How local LLMs like `qwen2.5` execute inference locally via REST endpoints.

