# Architectural Design: Bank-Agnostic Credit Card Insurance RAG System

**Date:** 2026-10-23  
**Status:** In Review / Ready for Implementation Planning  
**Target Card Benchmark:** TD Aeroplan Visa Infinite Privilege Card (52 pages, 10 insurance policies)  
**Expansion Scope:** Agnostic across all banks (`cibc/`, `scotia/`, `td/`, and future banks)  

---

## 1. Executive Summary & Goals

### 1.1 Objective
Build an efficient, highly reliable, and bank-agnostic Retrieval-Augmented Generation (RAG) system that allows users to select any credit card from a dropdown menu and ask detailed questions about its insurance coverage (e.g., travel medical, flight delay, rental car collision, mobile device protection).

### 1.2 Core Constraints & Principles
1. **Zero Bank-Specific Deterministic Rules:** The ingestion and retrieval engines must not rely on hardcoded bank regex or branching logic (e.g., `if bank == 'td'`). The architecture must handle heterogeneous formatting wholistically.
2. **Open-Source & Low-Cost Default:** Operates 100% locally by default using local embeddings and Ollama (e.g., Qwen 2.5 / Llama 3.2), with an optional pluggable toggle for cloud APIs (Groq, Gemini Flash, OpenAI).
3. **High Reliability & Grounding:** Insurance policies contain nuanced conditions, strict dollar caps, and exclusions (e.g., pre-existing medical conditions, trip length limits, age brackets). Answers must never hallucinate coverage; every claim must cite exact certificate sections and page numbers.
4. **Full-Stack Experience:** A clean, responsive web interface (FastAPI backend + Vanilla HTML/CSS/JS frontend) featuring card selection, streaming responses, and an interactive source citation drawer.

---

## 2. Analysis of Architectural Approaches & Selection Rationale

We evaluated three architectural paradigms against the core requirements.

| Evaluation Dimension | Approach 1: Hierarchical Layout-Aware + Breadcrumbs + Hybrid Retrieval (Selected) | Approach 2: LLM-Driven Offline Schema Distillation | Approach 3: Two-Stage Policy Routing + Deep Search |
| :--- | :--- | :--- | :--- |
| **Bank Agnosticism** | **High**: Relies only on universal PDF layout structures (headings, tables, font hierarchy). | **Medium**: Requires defining rigid schemas that might not fit new or unusual insurance riders. | **Medium**: Requires classifying queries into fixed policy categories upfront. |
| **Zero Deterministic Rules** | **Yes**: Document structure is parsed dynamically from layout markers. | **Yes**: LLMs extract the data, but prompts must generalize across diverse schemas. | **Partial**: Routing classifiers often struggle with overlapping or multi-policy queries. |
| **Ingestion Cost & Latency** | **Fast & Free**: 10–30 seconds per 50-page PDF on local CPU; no LLM needed during ingestion. | **Slow & Heavy**: Requires LLM to read 50+ pages per document. On CPU/Ollama, this takes hours. | **Fast**: Standard embedding generation per document. |
| **Fine-Print & Exclusion Catching** | **High**: Breadcrumb context keeps exclusions tied to their parent benefit; BM25 catches exact phrases. | **High Risk**: If the offline LLM drops a single exclusion clause during batching, it is lost forever. | **Medium**: Risk of routing to the wrong policy and missing cross-cutting clauses. |
| **Multi-Benefit Queries** | **Seamless**: Hybrid search surfaces clauses from multiple benefits simultaneously. | **Good**: Can query multiple relational tables/JSON fields. | **Poor**: Single-policy routers fail on queries like *"My trip was cancelled because I got sick in hospital, what can I claim?"* |
| **Runtime Infrastructure** | **Lightweight**: Local vector index + BM25 index + local embedding model. | **Heavy**: Structured database (SQLite/PostgreSQL) + Vector Store + complex syncing pipeline. | **Medium**: Multiple vector collections or complex routing index. |

### 2.1 Why Approach 1 Was Chosen (Pros & Cons Breakdown)

#### Approach 1: Hierarchical Layout-Aware Ingestion + Breadcrumb Chunking + Hybrid Retrieval (Selected)
* **Pros:**
  - **Truly Bank-Agnostic:** Every bank's insurance PDF follows visual document hierarchies (Title $\rightarrow$ Policy Heading $\rightarrow$ Section Heading $\rightarrow$ Body / Tables). Layout-aware parsers (`pymupdf4llm`) turn visual layouts into structured Markdown without bank-specific regex.
  - **Preserves Critical Context:** Naive chunking detaches exclusion clauses from their parent policy. By prepending hierarchical breadcrumbs (`[Card: TD Aeroplan VIP] > [Benefit: Flight/Trip Delay] > [Section: Limitations and Exclusions] (Page 33)`), every chunk retains full contextual awareness.
  - **High Recall via Hybrid Search:** Pure vector search frequently misses exact numbers (e.g., "$1,000", "48 days") or specific medical terms ("angina", "pre-existing"). BM25 lexical search combined with dense vectors via Reciprocal Rank Fusion (RRF) guarantees both semantic and exact keyword matches.
  - **Lightweight & Instant Ingestion:** A new card's PDF can be dropped into `data/<bank>/` and indexed in seconds on any standard laptop without incurring API fees or burning local GPU/CPU cycles.
* **Cons & Mitigations:**
  - *Risk:* PDFs with irregular layout or non-standard heading tags might lose hierarchy.
  - *Mitigation:* We use a fallback heuristic based on font size and bold markers in `pymupdf` to infer Markdown headers (`#`, `##`, `###`) if structural bookmarks are absent.

#### Approach 2: LLM-Driven Offline Schema Distillation (Rejected)
* **Pros:** Answering specific limit questions (e.g., *"What is the car rental MSRP limit?"*) becomes an instantaneous SQL/JSON lookup.
* **Cons:**
  - High risk of extraction hallucination: If an extraction prompt misses a small caveat (e.g., *"Coverage reduces to 3 days if age 65+"*), all future user queries will receive wrong answers.
  - Ingestion throughput is extremely slow on local hardware without costly cloud LLM tokens.
  - Violates the YAGNI principle for our core use case.

#### Approach 3: Two-Stage Policy Routing + Deep Search (Rejected)
* **Pros:** Narrows search space to a single sub-document, reducing noise.
* **Cons:** Real-world insurance queries often span multiple policies (e.g., Flight Delay vs Trip Interruption vs Emergency Medical). Hard routing creates false negatives and brittle boundaries.

---

## 3. Detailed System Architecture

```
┌────────────────────────────────────────────────────────────────────────┐
│                        Web Application (Frontend)                      │
│   • Bank & Card Dropdown (auto-discovered from filesystem)            │
│   • Streaming Chat Window with Markdown rendering                     │
│   • Citation Inspector Drawer (collapsible with page previews)         │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ HTTP / Server-Sent Events (SSE)
┌───────────────────────────────────▼────────────────────────────────────┐
│                       FastAPI Application Server                       │
│  ┌─────────────────────────┐               ┌────────────────────────┐  │
│  │  Card & Bank Registry   │               │   Query Orchestrator   │  │
│  │  • Scans data/*/*.pdf   │               │  • Hybrid Query Engine │  │
│  │  • Card ID normalization│               │  • Context Assembly    │  │
│  └───────────┬─────────────┘               └───────────┬────────────┘  │
│              │                                         │               │
│  ┌───────────▼─────────────────────────────────────────▼────────────┐  │
│  │                    Card-Scoped Storage Engine                    │  │
│  │  • BM25 Lexical Index (per card / memory-mapped)                 │  │
│  │  • Dense Vector Index (Qdrant / local FAISS / numpy store)       │  │
│  │  • Chunk Metadata Store (page numbers, section paths, text)      │  │
│  └─────────────────────────────────────────────────────┬────────────┘  │
│                                                        │ Top K Chunks  │
│  ┌─────────────────────────────────────────────────────▼────────────┐  │
│  │                     Pluggable LLM Generator                      │  │
│  │  • Local Ollama (qwen2.5:7b / llama3.2:latest)                   │  │
│  │  • Cloud API Toggle (Groq / Gemini / OpenAI compatible)          │  │
│  │  • Strict System Prompt for Grounding & Exclusions               │  │
│  └──────────────────────────────────────────────────────────────────┘  │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
┌───────────────────────────────────▼────────────────────────────────────┐
│                    Layout-Aware Ingestion Pipeline                     │
│  • Step 1: Layout extraction via pymupdf4llm (Markdown + Tables)       │
│  • Step 2: Hierarchical Section Tree Construction                      │
│  • Step 3: Breadcrumb Context Injection & Semantic Chunking            │
│  • Step 4: Local Embedding & BM25 Index Generation                     │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Ingestion & Breadcrumb Chunking Pipeline

### 4.1 Step 1: Layout-Aware Parsing
* Ingests digital PDFs using `pymupdf4llm` to preserve:
  - Document headers (`#`, `##`, `###`)
  - Markdown formatted tables (critical for Benefit Schedules and Limits)
  - Bolded terms and numbered lists
  - Exact source page numbers associated with each text block

### 4.2 Step 2: Hierarchical Tree Construction
* As the document is parsed, a stateful parser maintains the current navigation stack:
  ```
  Document: TD Aeroplan Visa Infinite Privilege Card
    └── Policy: FLIGHT/TRIP DELAY INSURANCE
          └── Section: Section 5 - Description of Insurance Coverage
                └── Subsection: Eligible Expenses (Page 33)
  ```

### 4.3 Step 3: Breadcrumb Context Injection
* For each chunk $C_i$, the breadcrumb path is prepended to the chunk text before embedding:
  ```markdown
  [Card: TD Aeroplan VIP] > [Policy: Flight/Trip Delay] > [Section: Description of Coverage] (Page 33)
  --------------------------------------------------
  The Insurer will reimburse reasonable expenses up to $1,000 per insured person for 
  necessary meals and accommodation if the flight is delayed for more than 4 hours...
  ```
* **Why this prevents hallucinations:** Even if the chunk only discusses "meals and accommodation up to $1,000", the embedding and the LLM explicitly see that this applies to *Flight/Trip Delay* on the *TD Aeroplan VIP* card, preventing false matches against other policies.

### 4.4 Step 4: Chunk Sizing Strategy
* **Target Size:** 400–600 tokens with 100-token overlap.
* **Boundary Rules:** Chunks never split mid-sentence or mid-table. Tables under 400 tokens are preserved as atomic chunks.

---

## 5. Retrieval & Re-ranking Engine

### 5.1 Card-Scoped Hybrid Search
When a user asks a question with a selected card:
1. **Filtering Scope:** Search is strictly isolated to the selected `card_id` index.
2. **Dense Vector Retrieval:**
   - Model: `sentence-transformers/all-MiniLM-L6-v2` (or `BAAI/bge-small-en-v1.5`), executing fast local CPU embeddings.
   - Computes cosine similarity against the card's dense vector index. Top 15 results retrieved.
3. **BM25 Lexical Retrieval:**
   - Tokenizes the query and retrieves top 15 results from the card's BM25 index.
   - Accurately captures exact numbers, currency figures, and specialized insurance terminology.
4. **Reciprocal Rank Fusion (RRF):**
   $$RRF(d) = \sum_{m \in \{dense, bm25\}} \frac{1}{60 + \text{rank}_m(d)}$$
   Produces a unified ranked candidate list of top 10 chunks.

### 5.2 Cross-Encoder Re-Ranking (Lightweight)
* Pass top 10 candidates through a lightweight cross-encoder (e.g. `cross-encoder/ms-marco-MiniLM-L-6-v2`).
* Ranks the top 4–5 most relevant chunks to send to the LLM context, ensuring exclusion and limitation clauses rank above superficial mentions.

---

## 6. Synthesis Engine & Structured Grounding Guardrails

### 6.1 System Prompt Design
The prompt strictly enforces grounded insurance advising:
```text
You are an expert Credit Card Insurance Specialist. Your goal is to provide accurate, reliable, 
and clear explanations of credit card insurance benefits strictly based on the provided context.

RULES:
1. ONLY answer based on the provided certificate context. Do not guess or assume.
2. If the user asks about coverage, ALWAYS state the direct answer, the coverage limits/durations, 
   and any mandatory conditions or exclusions (e.g., age limits, charging requirements, pre-existing conditions).
3. If an exclusion or limitation is mentioned in the context that applies to the query, you MUST highlight it.
4. If the context does not contain enough information to answer, state clearly that the policy document does not specify it.
5. Provide citations with exact Certificate Section and Page numbers in the format: [Section Name, Page X].

RESPONSE FORMAT:
- Direct Answer: Direct, concise summary of coverage.
- Coverage Limits & Details: Specific dollar amounts, allowable days, covered persons.
- Key Conditions & Exclusions: Specific criteria required to qualify, exclusions, or fine print.
- Sources & Citations: [Policy Name, Section, Page X]
```

### 6.2 Pluggable LLM Backend
* **Local Default:** Calls Ollama via `http://localhost:11434/api/generate` (using `qwen2.5:7b` or `llama3.2:latest`).
* **Cloud API Support:** Configurable via `.env` (`LLM_PROVIDER=ollama|groq|gemini|openai`).

---

## 7. Web Application & API Specification

### 7.1 Backend API Endpoints (FastAPI)
* `GET /api/cards`: Returns list of all detected banks and cards:
  ```json
  [
    {
      "id": "td_aeroplan_vip",
      "bank": "TD",
      "card_name": "TD Aeroplan Visa Infinite Privilege Card",
      "file_path": "data/td/TD Aeroplan Visa Infinite Privilege Card COI.pdf",
      "indexed": true
    }
  ]
  ```
* `POST /api/query`:
  - Request: `{"card_id": "td_aeroplan_vip", "question": "Does my card cover flight delays?"}`
  - Response (Streaming SSE or JSON): Returns answer, structured sections, and citation snippets with page numbers.
* `POST /api/ingest`: Forces re-indexing or initial ingestion of a card PDF.

### 7.2 Frontend Web Interface (Vanilla HTML5 / Modern CSS / Vanilla JS)
* **Header / Navigation:** Premium dark/modern aesthetic with bank selector and card dropdown.
* **Interactive Chat Panel:** Real-time conversation stream with formatted Markdown, alert callouts for exclusions, and badge tags for limits.
* **Collapsible Source Drawer:** Displays the exact text chunks retrieved from the PDF, highlighting the page number, section header, and similarity score.

---

## 8. Verification & Testing Strategy

### 8.1 Benchmark Test Cases (TD Aeroplan Visa Infinite Privilege)
We will validate the system against 10 critical insurance queries:
1. **Flight Delay:** Minimum delay hours required for reimbursement and maximum meal/hotel allowance. (Ground truth: 4 hours delay, up to $1,000, Section 5, Page 33).
2. **Emergency Medical Pre-Existing Conditions:** Stability period required for pre-existing medical conditions before departure. (Ground truth: Stability requirements based on age, Section 6, Pages 7–8).
3. **Auto Rental MSRP Cap:** Maximum Manufacturer's Suggested Retail Price (MSRP) of eligible rental cars. (Ground truth: Up to $85,000, Section 4, Page 36).
4. **Delayed Baggage:** Minimum delay threshold before emergency purchase of clothing/toiletries is covered. (Ground truth: 4 hours delay, up to $1,000, Section 5, Page 29).
5. **Mobile Device Insurance:** Deductible and maximum coverage limit per occurrence. (Ground truth: Up to $1,500 with 10% deductible, Section 2, Page 45).
6. **Age Cutoff for Travel Medical:** Duration of coverage for insured persons under 65 vs 65 and older. (Ground truth: 31 days for <65, 4 days for 65+, Section 1 & 5, Pages 2 & 5).
7. **Trip Cancellation Conditions:** What qualifies as a covered cause (e.g., sudden illness vs voluntary change of mind). (Ground truth: Covered sickness/injury vs explicit exclusion for change of plans, Section 5 & 6, Pages 16–18).
8. **Purchase Security & Extended Warranty:** Duration of duplicate warranty and claim window after incident. (Ground truth: Doubles manufacturer warranty up to 2 additional years, Section 3, Page 40).
9. **Hotel/Motel Burglary:** Coverage limit and conditions (forceful entry requirement). (Ground truth: Up to $2,500, visible marks of forced entry, Section 2, Page 50).
10. **Uncovered Negative Query:** *"Does this card insure pet health emergencies during travel?"* (Ground truth: Explicitly states that pet health is not covered under the certificate).

### 8.2 Evaluation Metrics
- **Recall@K:** Does the hybrid retrieval retrieve the exact section and page for all 10 benchmark queries? (Target: 100%).
- **Exclusion Capture Rate:** Does the generated response explicitly mention the exclusions and limits for conditional queries? (Target: 100%).
- **Citation Precision:** Does every cited page number match the actual page in the official PDF? (Target: 100%).
