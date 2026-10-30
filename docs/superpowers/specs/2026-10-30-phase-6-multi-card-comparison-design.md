# Architecture Design Spec: Phase 6 Multi-Card Policy Comparison Engine

**Date:** 2026-10-30  
**Phase:** Phase 6 — Multi-Card Policy Comparison Engine  
**Status:** Approved by User / Ready for Implementation Planning  
**Target Card Benchmark:** TD Aeroplan VIP vs CIBC Aeroplan VIP vs Scotiabank Passport VIP  
**Hardware & Stack:** Qdrant Cloud Hybrid Search (768-d Dense + BM25 Sparse, RRF k=60), Local Ollama (`qwen2.5:7b`), FastAPI SSE, Vanilla HTML5/CSS3/ES6 Dark Fintech UI  

---

## 1. Executive Summary & Objective

### 1.1 The Challenge
Credit card insurance holders and financial researchers frequently need to compare coverage limits, eligibility requirements, and strict exclusions between different premium credit cards (for instance, *TD Aeroplan Visa Infinite Privilege* vs *CIBC Aeroplan Visa Infinite Privilege* vs *Scotiabank Passport Visa Infinite Privilege*).
Currently, the system is strictly single-card scoped: the user selects one active card in the UI and queries it in isolation. Comparing two or three cards requires repetitive, manual queries, mental cross-referencing, and lacks an automated synthesis engine to contrast coverage figures, deductibles, and hidden conditions.

### 1.2 The Solution
Phase 6 delivers a fully integrated **Multi-Card Policy Comparison Engine**:
1. **Multi-Card Parallel Retrieval Pipeline (`src/retriever.py`)**: Fan-out / fan-in hybrid retrieval across 2 or 3 selected card booklets simultaneously, guaranteeing isolated search per card with zero cross-card vector dilution or card starvation.
2. **Comparative Synthesis Prompt (`src/generator.py`)**: Structured comparative prompt instructing local Ollama (`qwen2.5:7b`) to synthesize an **Executive Summary**, a **Side-by-Side Comparison Matrix (Markdown Table)**, **Critical Exclusions & Qualifying Conditions**, and **Verified Policy Citations**.
3. **SSE Streaming Comparison Endpoint (`api/main.py`)**: `POST /api/compare` endpoint streaming real-time tokens along with structured per-card citations and breadcrumbs.
4. **Side-by-Side Policy Matrix UI (`frontend/`)**: A dedicated comparison view switchable via header tabs, multi-card selection chips (supporting 2–3 cards), streaming markdown table rendering with dark fintech styling, and per-bank segmented citation accordions.
5. **Comprehensive 5-Rider Cross-Bank Benchmark (`tests/test_comparison.py`)**: Codified ground-truth evaluations across Rental Car Collision, Emergency Medical Out-of-Province Duration, Trip Cancellation / Interruption, Baggage Delay, and Mobile Device Protection.
6. **Command-Line CLI Tool (`scripts/compare_cards.py`)**: Standalone terminal client for running comparative benchmarks.

---

## 2. Core Architectural Principles & Invariants

1. **Card Context Isolation**:
   - Retrieval for each selected card runs in an isolated scope. Card A's dense or sparse vectors never compete against Card B's vectors during search.
   - Each card is allocated an equal quota of top `ParentSection`s (2 per card) and `ChildChunk` citations (4–6 per card), preventing one bank from starving another in the context prompt.
2. **Zero-Hallucination Comparative Grounding**:
   - The LLM is explicitly forbidden from inventing coverage for a card whose retrieved sections do not mention that rider. If a rider is omitted in the retrieved context for Card B, the matrix must explicitly state `"Not specified in retrieved policy sections"`.
3. **Identical Streaming Experience**:
   - The comparison endpoint mirrors `/api/query` in streaming latency and reliability via Server-Sent Events (SSE).
4. **Deterministic Offline Fallback**:
   - If Ollama is offline or times out, the backend generates a deterministic comparison markdown table directly from the retrieved parent section excerpts without crashing.
5. **Rigid Table Structure**:
   - The LLM is instructed to generate Markdown tables with columns:
     `| Coverage Rider / Benefit | [Card 1] | [Card 2] | [Card 3 (if 3 cards)] | Advantage / Key Distinction |`

---

## 3. Data Models & Backend Architecture

### 3.1 Data Structures in `src/retriever.py`

```python
from dataclasses import dataclass
from typing import Dict, List, Any
from src.parser import ParentSection

@dataclass
class MultiCardContext:
    card_contexts: Dict[str, RetrievedContext]  # card_id -> RetrievedContext
    card_names: Dict[str, str]                 # card_id -> display card name
```

### 3.2 Parallel Fan-Out Retriever (`src/retriever.py`)

Add `retrieve_multicard` method to `HybridParentRetriever`:

```python
def retrieve_multicard(
    self,
    card_ids: List[str],
    query: str,
    top_k_parents_per_card: int = 2,
    top_k_children_per_card: int = 6
) -> MultiCardContext:
    """Executes parallel hybrid search across multiple cards in Qdrant Cloud."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    card_contexts: Dict[str, RetrievedContext] = {}
    card_names: Dict[str, str] = {}

    def _fetch_card(cid: str):
        ctx = self.retrieve(
            card_id=cid,
            query=query,
            top_k_parents=top_k_parents_per_card,
            top_k_children=top_k_children_per_card
        )
        c_name = ctx.parents[0].card_name if ctx.parents else cid
        return cid, c_name, ctx

    with ThreadPoolExecutor(max_workers=min(len(card_ids), 3)) as executor:
        futures = [executor.submit(_fetch_card, cid) for cid in card_ids]
        for f in as_completed(futures):
            cid, c_name, ctx = f.result()
            card_contexts[cid] = ctx
            card_names[cid] = c_name

    return MultiCardContext(card_contexts=card_contexts, card_names=card_names)
```

### 3.3 Comparative Synthesis Prompt (`src/generator.py`)

```python
COMPARISON_SYSTEM_PROMPT = """You are an expert Credit Card Insurance Comparative Specialist.
Your task is to provide an objective, rigorous, and accurate side-by-side comparison of insurance coverage 
across the specified credit cards strictly based on their respective policy excerpts.

RULES:
1. ONLY use the provided context for each card. Do not assume or hallucinate benefits.
2. If a card's excerpt does not state a limit, condition, or rider, write "Not specified in retrieved policy sections".
3. Highlight exact numbers, dollar caps ($), day durations, age cutoffs, and required card charging percentages.
4. Structure your response using these EXACT headings:

### Executive Summary
[Direct, clear summary highlighting the most critical differences and the overall standout card for this benefit]

### Side-by-Side Comparison Matrix
| Coverage Rider / Benefit | [Card 1 Name] | [Card 2 Name] | [Card 3 Name if applicable] | Advantage / Winner |
| :--- | :--- | :--- | :--- | :--- |
| ... | ... | ... | ... | ... |

### Critical Exclusions & Qualifying Conditions
- **[Card 1 Name]**: [Stability requirements, charging %, age limits, deductibles]
- **[Card 2 Name]**: [...]

### Verified Policy Citations
- **[Card 1 Name]**: [Section Title, Page X]
- **[Card 2 Name]**: [...]
"""
```

Prompt assembly function:
```python
def _build_multicard_context_prompt(
    self,
    question: str,
    card_contexts: Dict[str, RetrievedContext],
    card_names: Dict[str, str]
) -> str:
    sections = []
    for cid, ctx in card_contexts.items():
        name = card_names.get(cid, cid)
        card_block = [f"=== CARD: {name} (ID: {cid}) ==="]
        for i, p in enumerate(ctx.parents):
            page_str = f"Page {p.page_start}" if p.page_start == p.page_end else f"Pages {p.page_start}-{p.page_end}"
            card_block.append(f"--- Parent Section {i+1}: [{p.policy_name}] > {p.section_title} ({page_str}) ---\n{p.content}\n")
        sections.append("\n".join(card_block))

    all_context = "\n\n".join(sections)
    return f"CONTEXT INFORMATION PER CARD:\n{all_context}\n\nCOMPARATIVE QUESTION:\n{question}\n\nSynthesize the comparison according to the rules and structure:"
```

### 3.4 API Layer (`api/main.py`)

```python
class CompareRequest(BaseModel):
    card_ids: List[str]
    query: str

@app.post("/api/compare")
def stream_compare(req: CompareRequest):
    if len(req.card_ids) < 2 or len(req.card_ids) > 3:
        raise HTTPException(status_code=400, detail="Comparison requires between 2 and 3 cards.")

    # Validate and auto-index any unindexed cards
    resolved_ids = []
    for cid in req.card_ids:
        eff_id = CARD_ALIASES.get(cid, cid)
        if not (qdrant_storage.is_indexed(eff_id) or legacy_storage.is_indexed(eff_id)):
            index_card(eff_id)
        resolved_ids.append(eff_id)

    return StreamingResponse(
        sse_compare_event_generator(resolved_ids, req.query),
        media_type="text/event-stream"
    )
```

SSE event stream emits:
1. `event: cards_context` — JSON with per-card snippets, breadcrumbs, and pages.
2. `event: token` — JSON string of streamed tokens.
3. `event: citations` — JSON with citations grouped by `card_id`.
4. `event: done` — JSON with `elapsed_ms`.

---

## 4. Frontend UI & Interaction Design

### 4.1 Header Mode Switcher (`frontend/index.html`)
A segmented tab pill in the top navigation:
```html
<div class="view-mode-tabs" role="tablist">
    <button class="mode-tab active" id="singleCardModeBtn" role="tab" aria-selected="true">
        <span>💬 Single Card Chat</span>
    </button>
    <button class="mode-tab" id="compareModeBtn" role="tab" aria-selected="false">
        <span>⚖️ Multi-Card Comparison</span>
    </button>
</div>
```

### 4.2 Multi-Card Selector & Chips
When in **Comparison Mode**:
- Single-card dropdown is replaced with a **Multi-Card Selector Container**.
- Cards are displayed as toggleable pills with checkmarks:
  `[✓ TD Aeroplan VIP]`, `[✓ CIBC Aeroplan VIP]`, `[+ Scotiabank Passport VIP]`.
- Enforces minimum 2 cards, maximum 3 cards. If 1 card is selected, displays hint: `Select at least one more card to compare`.

### 4.3 Side-by-Side Matrix Table CSS Styling (`frontend/style.css`)
- Tables rendered from markdown inside messages get automatic styling:
  - Container `.matrix-table-wrapper` with horizontal scroll and customized sleek dark scrollbars.
  - `#0F172A` header background with cyan text borders.
  - Alternating row background (`#0A0F1D` / `#0D1526`).
  - `.winner-col` or advantage cell highlighting with subtle emerald glow `#10B981`.

### 4.4 Per-Bank Citations Drawer
In comparison mode, the right-hand sources panel renders segmented accordion sections:
```
▼ TD Aeroplan Visa Infinite Privilege (2 Citations)
  • [Flight Delay Insurance > Benefits, Page 33]
  • [Rental Car Collision Damage > Types Covered, Page 38]
▶ CIBC Aeroplan Visa Infinite Privilege (2 Citations)
▶ Scotiabank Passport Visa Infinite Privilege (2 Citations)
```

---

## 5. Comprehensive 5-Rider Cross-Bank Ground Truth Benchmark

Test file: `tests/test_comparison.py`

| # | Comparison Rider | Query | TD Aeroplan VIP Expected | CIBC Aeroplan VIP Expected | Scotiabank Passport VIP Expected | Key Differentiating Winner |
| :- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1** | **Rental Car Collision Waiver** | "Compare maximum rental car MSRP coverage limit and rental duration." | `$85,000` MSRP / 48 consecutive days | `$100,000` MSRP / 48 consecutive days | `$65,000` MSRP / 31 consecutive days | **CIBC** ($100k MSRP) |
| **2** | **Emergency Travel Medical (Age 65+)** | "Compare out-of-province emergency medical coverage duration for age 65 and older." | `4 consecutive days` | `10 consecutive days` | `3 consecutive days` | **CIBC** (10 days) |
| **3** | **Trip Cancellation / Interruption** | "Compare trip cancellation maximum reimbursement limit per insured traveler." | `$2,500` per traveler (max $5,000) | `$2,500` per traveler (max $5,000) | `$2,500` per traveler (max $10,000) | **Scotia** ($10k trip cap) |
| **4** | **Baggage Delay Allowance** | "Compare baggage delay minimum hours requirement and maximum essential item coverage." | `4 hours` delay / `$1,000` limit | `4 hours` delay / `$1,000` limit | `4 hours` delay / `$1,000` limit | **Tie** across all 3 |
| **5** | **Mobile Device Protection** | "Compare mobile device insurance maximum coverage limit and deductible." | `$1,500` limit / 10% deductible (min $25) | `$1,500` limit / 10% deductible (min $25) | `$1,000` limit / 10% deductible (min $25) | **TD / CIBC** ($1,500) |

The test suite asserts:
1. `retrieve_multicard` returns valid `RetrievedContext` for all 3 cards with > 0 parents and citations.
2. Retrieved parents for each respective card contain the ground-truth terms and figures without contamination.
3. The generator produces a valid markdown comparison table containing column headers for all compared cards.

---

## 6. Implementation Phases & File Plan

1. **`src/retriever.py`**:
   - Add `MultiCardContext` dataclass.
   - Add `retrieve_multicard(card_ids, query)` with parallel `ThreadPoolExecutor` execution and card context mapping.
2. **`src/generator.py`**:
   - Add `COMPARISON_SYSTEM_PROMPT`.
   - Add `_build_multicard_context_prompt`, `_format_fallback_comparison`, and `generate_comparison_stream`.
3. **`api/main.py`**:
   - Add `CompareRequest` Pydantic model.
   - Add `sse_compare_event_generator` and `POST /api/compare` endpoint.
4. **`frontend/`**:
   - `index.html`: Add mode switcher tabs and multi-card selector bar.
   - `style.css`: Add styles for mode tabs, multi-card pills, comparison matrix table, and segmented citation drawers.
   - `app.js`: Implement comparison mode state, multi-card selection logic, `/api/compare` SSE streaming consumer, markdown table parser, and segmented citation rendering.
5. **`scripts/compare_cards.py`**:
   - Terminal CLI for cross-card comparison queries with streaming output.
6. **`tests/test_comparison.py`**:
   - Automated unit tests for multi-card retriever, generator, API endpoint, and the 5-rider benchmark across TD vs CIBC vs Scotiabank.
7. **Documentation**:
   - Create `docs/phases/PHASE_6_COMPARISON.md`.
   - Update `docs/ROADMAP.md` and `README.md` to reflect Phase 6 completion.
