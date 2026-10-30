# Phase 6 Multi-Card Policy Comparison Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a multi-card policy comparison engine that retrieves coverage clauses across 2–3 credit card insurance booklets simultaneously, synthesizes a side-by-side comparative matrix via local LLM / Ollama with strict citation guardrails, exposes a streaming `POST /api/compare` endpoint, and provides an interactive dark fintech UI matrix view.

**Architecture:** Parallel fan-out hybrid retrieval (`HybridParentRetriever.retrieve_multicard`) queries Qdrant Cloud per card independently using ThreadPoolExecutor to prevent vector starvation. The resulting `MultiCardContext` is formatted into card-isolated sections for `LLMGenerator`, which prompts `qwen2.5:7b` to emit an Executive Summary, Side-by-Side Markdown Comparison Matrix, Critical Exclusions, and Verified Policy Citations. FastAPI streams tokens and structured citations via SSE. The frontend toggles between Single Card Chat and Multi-Card Comparison Mode with responsive matrix tables and segmented per-bank citation accordions.

**Tech Stack:** Python 3.13, FastAPI, Uvicorn, httpx, Qdrant Cloud, sentence-transformers (`BAAI/bge-base-en-v1.5`), BM25Plus, Ollama (`qwen2.5:7b`), Vanilla HTML5/CSS3/ES6 (Zero build step), pytest.

**Spec:** [docs/superpowers/specs/2026-10-30-phase-6-multi-card-comparison-design.md](file:///d:/rag-practice/docs/superpowers/specs/2026-10-30-phase-6-multi-card-comparison-design.md)

## Global Constraints

- Never use bank-specific regex or deterministic `if bank == 'td'` hardcoding in parsing or core search logic.
- Card context isolation: Each compared card must have an isolated retrieval scope; search scores from Card A must never crowd out Card B.
- Zero-hallucination comparative grounding: If a card does not specify a benefit or limit in the retrieved context, write "Not specified in retrieved policy sections".
- Streaming consistency: `POST /api/compare` must stream SSE events (`cards_context`, `token`, `citations`, `done`) matching `/api/query`.
- Deterministic offline resilience: Fallback generator must output a markdown comparison table if Ollama is offline.
- Pure vanilla frontend: No npm build steps, Webpack, or external JS frameworks.

---

### Task 1: Multi-Card Parallel Hybrid Retriever (`src/retriever.py`)

**Files:**
- Modify: `src/retriever.py:10-35`
- Test: `tests/test_comparison.py`

**Interfaces:**
- Consumes: `HybridParentRetriever.retrieve(card_id: str, query: str, top_k_parents: int, top_k_children: int) -> RetrievedContext`
- Produces: `MultiCardContext` dataclass and `retrieve_multicard(self, card_ids: List[str], query: str, top_k_parents_per_card: int = 2, top_k_children_per_card: int = 6) -> MultiCardContext`

- [ ] **Step 1: Write the failing unit test for `retrieve_multicard`**

Create `tests/test_comparison.py`:
```python
from unittest.mock import MagicMock
import pytest
from src.parser import ParentSection
from src.retriever import HybridParentRetriever, RetrievedContext, MultiCardContext

def test_retriever_multicard_parallel_fanout():
    mock_storage = MagicMock()
    retriever = HybridParentRetriever(storage_manager=mock_storage)

    # Mock retrieve method to return card-specific parents
    def fake_retrieve(card_id: str, query: str, top_k_parents: int = 2, top_k_children: int = 6):
        return RetrievedContext(
            parents=[ParentSection(
                parent_id=f"{card_id}_p1",
                card_id=card_id,
                card_name=f"Card Name for {card_id}",
                policy_name="Flight Delay",
                section_title="Benefits",
                page_start=10,
                page_end=11,
                content=f"Coverage content for {card_id}"
            )],
            citations=[{"child_id": f"{card_id}_c1", "page_number": 10, "breadcrumb": "Flight Delay", "content": "Excerpt", "score": 0.85}]
        )

    retriever.retrieve = MagicMock(side_effect=fake_retrieve)

    card_ids = ["td_aeroplan_vip", "cibc_aeroplan_vip", "scotia_passport_vip"]
    result = retriever.retrieve_multicard(card_ids=card_ids, query="What is the flight delay coverage limit?")

    assert isinstance(result, MultiCardContext)
    assert len(result.card_contexts) == 3
    assert len(result.card_names) == 3
    for cid in card_ids:
        assert cid in result.card_contexts
        ctx = result.card_contexts[cid]
        assert len(ctx.parents) == 1
        assert ctx.parents[0].card_id == cid
        assert cid in result.card_names
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_comparison.py::test_retriever_multicard_parallel_fanout -v`  
Expected: FAIL with `ImportError: cannot import name 'MultiCardContext'` or `AttributeError: 'HybridParentRetriever' object has no attribute 'retrieve_multicard'`.

- [ ] **Step 3: Implement `MultiCardContext` and `retrieve_multicard` in `src/retriever.py`**

In `src/retriever.py`:
```python
from concurrent.futures import ThreadPoolExecutor, as_completed

@dataclass
class MultiCardContext:
    card_contexts: Dict[str, RetrievedContext]  # card_id -> RetrievedContext
    card_names: Dict[str, str]                 # card_id -> display card name
```

Add inside `HybridParentRetriever`:
```python
    def retrieve_multicard(
        self,
        card_ids: List[str],
        query: str,
        top_k_parents_per_card: int = 2,
        top_k_children_per_card: int = 6
    ) -> MultiCardContext:
        """Executes parallel hybrid search across multiple cards in Qdrant Cloud."""
        card_contexts: Dict[str, RetrievedContext] = {}
        card_names: Dict[str, str] = {}

        def _fetch_card(cid: str):
            ctx = self.retrieve(
                card_id=cid,
                query=query,
                top_k_parents=top_k_parents_per_card,
                top_k_children=top_k_children_per_card
            )
            c_name = ctx.parents[0].card_name if ctx.parents and ctx.parents[0].card_name else cid
            return cid, c_name, ctx

        with ThreadPoolExecutor(max_workers=min(len(card_ids), 3)) as executor:
            futures = [executor.submit(_fetch_card, cid) for cid in card_ids]
            for f in as_completed(futures):
                cid, c_name, ctx = f.result()
                card_contexts[cid] = ctx
                card_names[cid] = c_name

        return MultiCardContext(card_contexts=card_contexts, card_names=card_names)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_comparison.py::test_retriever_multicard_parallel_fanout -v`  
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/retriever.py tests/test_comparison.py
git commit -m "feat(retriever): implement parallel multi-card hybrid retrieval"
```

---

### Task 2: Comparative Synthesis Prompt & Generator Stream (`src/generator.py`)

**Files:**
- Modify: `src/generator.py`
- Test: `tests/test_comparison.py`

**Interfaces:**
- Consumes: `MultiCardContext` from `src/retriever.py`
- Produces:
  - `COMPARISON_SYSTEM_PROMPT: str`
  - `LLMGenerator._build_multicard_context_prompt(question: str, multi_context: MultiCardContext) -> str`
  - `LLMGenerator._format_fallback_comparison(question: str, multi_context: MultiCardContext) -> str`
  - `LLMGenerator.generate_comparison_stream(question: str, multi_context: MultiCardContext) -> Generator[str, None, None]`

- [ ] **Step 1: Write the failing tests for multi-card prompt building and fallback generation**

Add to `tests/test_comparison.py`:
```python
from src.generator import LLMGenerator, COMPARISON_SYSTEM_PROMPT

def test_generator_comparison_prompt_assembly():
    generator = LLMGenerator()
    multi_ctx = MultiCardContext(
        card_contexts={
            "card_a": RetrievedContext(
                parents=[ParentSection("p1", "card_a", "Bank A Card", "Rental Car", "Limits", 10, 10, "MSRP up to $85,000 for 48 days.")],
                citations=[]
            ),
            "card_b": RetrievedContext(
                parents=[ParentSection("p2", "card_b", "Bank B Card", "Rental Car", "Limits", 20, 20, "MSRP up to $100,000 for 48 days.")],
                citations=[]
            )
        },
        card_names={"card_a": "Bank A Card", "card_b": "Bank B Card"}
    )

    prompt = generator._build_multicard_context_prompt("Compare rental car limits", multi_ctx)
    assert "=== CARD: Bank A Card (ID: card_a) ===" in prompt
    assert "=== CARD: Bank B Card (ID: card_b) ===" in prompt
    assert "MSRP up to $85,000" in prompt
    assert "MSRP up to $100,000" in prompt
    assert "COMPARATIVE QUESTION:\nCompare rental car limits" in prompt

def test_generator_comparison_fallback_markdown():
    generator = LLMGenerator()
    multi_ctx = MultiCardContext(
        card_contexts={
            "td_aeroplan_vip": RetrievedContext(
                parents=[ParentSection("p1", "td_aeroplan_vip", "TD Aeroplan VIP", "Rental Car", "Limits", 38, 38, "Maximum MSRP $85,000.")],
                citations=[]
            ),
            "cibc_aeroplan_vip": RetrievedContext(
                parents=[ParentSection("p2", "cibc_aeroplan_vip", "CIBC Aeroplan VIP", "Rental Car", "Limits", 24, 24, "Maximum MSRP $100,000.")],
                citations=[]
            )
        },
        card_names={"td_aeroplan_vip": "TD Aeroplan VIP", "cibc_aeroplan_vip": "CIBC Aeroplan VIP"}
    )

    fallback = generator._format_fallback_comparison("Compare rental car limits", multi_ctx)
    assert "### Executive Summary" in fallback
    assert "### Side-by-Side Comparison Matrix" in fallback
    assert "| Coverage Rider / Benefit |" in fallback
    assert "TD Aeroplan VIP" in fallback
    assert "CIBC Aeroplan VIP" in fallback
    assert "### Critical Exclusions & Qualifying Conditions" in fallback
    assert "### Verified Policy Citations" in fallback
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_comparison.py::test_generator_comparison_prompt_assembly tests/test_comparison.py::test_generator_comparison_fallback_markdown -v`  
Expected: FAIL with `AttributeError: 'LLMGenerator' object has no attribute '_build_multicard_context_prompt'`.

- [ ] **Step 3: Implement comparison prompt, fallback, and streaming generator in `src/generator.py`**

In `src/generator.py`:
```python
from src.retriever import ScoredChunk, RetrievedContext, MultiCardContext

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

Add methods to `LLMGenerator`:
```python
    def _build_multicard_context_prompt(self, question: str, multi_context: MultiCardContext) -> str:
        sections = []
        for cid, ctx in multi_context.card_contexts.items():
            name = multi_context.card_names.get(cid, cid)
            card_block = [f"=== CARD: {name} (ID: {cid}) ==="]
            for i, p in enumerate(ctx.parents):
                page_str = f"Page {p.page_start}" if p.page_start == p.page_end else f"Pages {p.page_start}-{p.page_end}"
                card_block.append(f"--- Parent Section {i+1}: [{p.policy_name}] > {p.section_title} ({page_str}) ---\n{p.content}\n")
            sections.append("\n".join(card_block))

        all_context = "\n\n".join(sections)
        return f"CONTEXT INFORMATION PER CARD:\n{all_context}\n\nCOMPARATIVE QUESTION:\n{question}\n\nSynthesize the comparison according to the rules and structure:"

    def _format_fallback_comparison(self, question: str, multi_context: MultiCardContext) -> str:
        card_names = list(multi_context.card_names.values())
        headers = ["Coverage Rider / Benefit"] + card_names + ["Advantage / Winner"]
        table_header = "| " + " | ".join(headers) + " |"
        table_separator = "| " + " | ".join([":---"] * len(headers)) + " |"

        rows = []
        rider_title = question.replace("Compare", "").replace("coverage", "").replace("limits", "").strip().title() or "Coverage Comparison"
        row_cols = [rider_title]
        for cid in multi_context.card_contexts.keys():
            ctx = multi_context.card_contexts[cid]
            if ctx.parents:
                p = ctx.parents[0]
                snippet = p.content[:120].replace("\n", " ").strip() + "..."
                row_cols.append(snippet)
            else:
                row_cols.append("Not specified in retrieved policy sections")
        row_cols.append("Refer to comparison matrix columns")
        rows.append("| " + " | ".join(row_cols) + " |")

        matrix_table = f"{table_header}\n{table_separator}\n" + "\n".join(rows)

        exclusion_lines = []
        citation_lines = []
        for cid, ctx in multi_context.card_contexts.items():
            name = multi_context.card_names.get(cid, cid)
            card_citations = []
            for p in ctx.parents:
                page_str = f"Page {p.page_start}" if p.page_start == p.page_end else f"Pages {p.page_start}-{p.page_end}"
                card_citations.append(f"[{p.section_title}, {page_str}]")
            citation_lines.append(f"- **{name}**: {', '.join(card_citations) if card_citations else 'No citations'}")
            exclusion_lines.append(f"- **{name}**: Refer to policy terms and conditions cited below.")

        return (
            f"### Executive Summary\nComparison synthesized from verified policy sections for '{question}'.\n\n"
            f"### Side-by-Side Comparison Matrix\n{matrix_table}\n\n"
            f"### Critical Exclusions & Qualifying Conditions\n" + "\n".join(exclusion_lines) + "\n\n"
            f"### Verified Policy Citations\n" + "\n".join(citation_lines)
        )

    def generate_comparison_stream(self, question: str, multi_context: MultiCardContext) -> Generator[str, None, None]:
        user_prompt = self._build_multicard_context_prompt(question, multi_context)

        if self.ollama_client:
            res = self.ollama_client.chat(model=self.model, messages=[
                {"role": "system", "content": COMPARISON_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ], stream=True)
            for part in res:
                if isinstance(part, dict) and "message" in part and "content" in part["message"]:
                    yield part["message"]["content"]
            return

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": COMPARISON_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            "stream": True
        }

        client_timeout = httpx.Timeout(120.0, connect=5.0)
        try:
            with httpx.stream("POST", f"{self.base_url}/api/chat", json=payload, timeout=client_timeout) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line:
                        continue
                    data = json.loads(line)
                    content = data.get("message", {}).get("content", "")
                    if content:
                        yield content
        except Exception:
            yield self._format_fallback_comparison(question, multi_context)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_comparison.py::test_generator_comparison_prompt_assembly tests/test_comparison.py::test_generator_comparison_fallback_markdown -v`  
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/generator.py tests/test_comparison.py
git commit -m "feat(generator): add comparative synthesis prompt and multi-card streaming generator"
```

---

### Task 3: FastAPI Multi-Card Comparison Endpoint (`api/main.py`)

**Files:**
- Modify: `api/main.py`
- Test: `tests/test_comparison.py`

**Interfaces:**
- Consumes: `CompareRequest(card_ids: List[str], query: str)`
- Produces: `POST /api/compare` returning SSE stream with `event: cards_context`, `event: token`, `event: citations`, `event: done`.

- [ ] **Step 1: Write failing tests for `/api/compare` validation and streaming**

Add to `tests/test_comparison.py`:
```python
from fastapi.testclient import TestClient
from api.main import app

api_client = TestClient(app)

def test_api_compare_validation_requires_2_or_3_cards():
    # Only 1 card
    res1 = api_client.post("/api/compare", json={"card_ids": ["td_aeroplan_vip"], "query": "Compare limits"})
    assert res1.status_code == 400
    assert "between 2 and 3 cards" in res1.json()["detail"]

    # 4 cards
    res2 = api_client.post("/api/compare", json={
        "card_ids": ["card_1", "card_2", "card_3", "card_4"],
        "query": "Compare limits"
    })
    assert res2.status_code == 400

def test_api_compare_streaming_events():
    payload = {
        "card_ids": ["td_aeroplan_vip", "cibc_aeroplan_vip"],
        "query": "Compare rental car collision limits"
    }
    with api_client.stream("POST", "/api/compare", json=payload) as response:
        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]
        events = []
        for line in response.iter_lines():
            if line:
                events.append(line)
        assert any("event: cards_context" in e for e in events)
        assert any("event: token" in e for e in events)
        assert any("event: citations" in e for e in events)
        assert any("event: done" in e for e in events)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_comparison.py::test_api_compare_validation_requires_2_or_3_cards tests/test_comparison.py::test_api_compare_streaming_events -v`  
Expected: FAIL with `404 Not Found` for `/api/compare`.

- [ ] **Step 3: Implement `CompareRequest` and `POST /api/compare` in `api/main.py`**

In `api/main.py`:
```python
class CompareRequest(BaseModel):
    card_ids: List[str]
    query: str

def sse_compare_event_generator(card_ids: List[str], query: str) -> Generator[str, None, None]:
    start_total = time.time()

    # Multi-card retrieval
    multi_ctx = qdrant_retriever.retrieve_multicard(
        card_ids=card_ids,
        query=query,
        top_k_parents_per_card=2,
        top_k_children_per_card=6
    )

    # 1. Emit cards_context event
    context_payload = {
        cid: [
            {
                "rank": i + 1,
                "page_number": c.get("page_number", 1),
                "breadcrumb": c.get("breadcrumb", ""),
                "snippet": c.get("content", "")[:200] + "..." if len(c.get("content", "")) > 200 else c.get("content", ""),
                "full_content": c.get("content", ""),
                "score": round(c.get("score", 0.0), 4)
            }
            for i, c in enumerate(ctx.citations)
        ]
        for cid, ctx in multi_ctx.card_contexts.items()
    }
    yield f"event: cards_context\ndata: {json.dumps(context_payload)}\n\n"

    # 2. Stream tokens
    token_stream = generator.generate_comparison_stream(query, multi_ctx)
    for token in token_stream:
        yield f"event: token\ndata: {json.dumps(token)}\n\n"

    # 3. Emit grouped citations
    citations_payload = {
        cid: ctx.citations
        for cid, ctx in multi_ctx.card_contexts.items()
    }
    yield f"event: citations\ndata: {json.dumps(citations_payload)}\n\n"

    elapsed = round((time.time() - start_total) * 1000, 1)
    yield f"event: done\ndata: {json.dumps({'elapsed_ms': elapsed})}\n\n"

@app.post("/api/compare")
def stream_compare(req: CompareRequest):
    if len(req.card_ids) < 2 or len(req.card_ids) > 3:
        raise HTTPException(status_code=400, detail="Comparison requires between 2 and 3 cards.")

    resolved_ids = []
    for cid in req.card_ids:
        eff_id = CARD_ALIASES.get(cid, cid)
        if not (qdrant_storage.is_indexed(eff_id) or legacy_storage.is_indexed(eff_id)):
            index_card(eff_id)
            if not (qdrant_storage.is_indexed(eff_id) or legacy_storage.is_indexed(eff_id)):
                eff_id = CARD_ALIASES.get(eff_id, eff_id)
        resolved_ids.append(eff_id)

    return StreamingResponse(
        sse_compare_event_generator(resolved_ids, req.query),
        media_type="text/event-stream"
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_comparison.py::test_api_compare_validation_requires_2_or_3_cards tests/test_comparison.py::test_api_compare_streaming_events -v`  
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add api/main.py tests/test_comparison.py
git commit -m "feat(api): add POST /api/compare SSE streaming endpoint"
```

---

### Task 4: Standalone Terminal CLI Utility (`scripts/compare_cards.py`)

**Files:**
- Create: `scripts/compare_cards.py`

**Interfaces:**
- Consumes: CLI flags `--cards [c1, c2, ...]` and `--query [text]`
- Produces: Terminal execution with streaming output and formatted ANSI comparison matrix.

- [ ] **Step 1: Create `scripts/compare_cards.py`**

```python
import argparse
import sys
import time
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.storage import QdrantStorageManager
from src.retriever import HybridParentRetriever
from src.generator import LLMGenerator

def main():
    parser = argparse.ArgumentParser(description="Multi-Card Insurance Policy Comparison Terminal Client")
    parser.add_argument("--cards", nargs="+", required=True, help="2 or 3 Card IDs to compare (e.g. td_aeroplan_vip cibc_aeroplan_vip)")
    parser.add_argument("--query", type=str, required=True, help="Comparison query (e.g. 'Compare rental car collision limits')")
    args = parser.parse_args()

    if len(args.cards) < 2 or len(args.cards) > 3:
        print("\033[91mError: Please specify 2 or 3 cards to compare.\033[0m")
        sys.exit(1)

    print(f"\n\033[96m========================================================================\033[0m")
    print(f"\033[96m  CREDIT CARD INSURANCE RAG — MULTI-CARD COMPARISON ENGINE\033[0m")
    print(f"\033[96m  Cards: {', '.join(args.cards)}\033[0m")
    print(f"\033[96m  Query: {args.query}\033[0m")
    print(f"\033[96m========================================================================\033[0m\n")

    storage = QdrantStorageManager()
    retriever = HybridParentRetriever(storage_manager=storage)
    generator = LLMGenerator()

    start_retrieval = time.time()
    print("\033[93m[1/2] Fan-out hybrid retrieval across card booklets...\033[0m")
    multi_ctx = retriever.retrieve_multicard(card_ids=args.cards, query=args.query)
    retrieval_ms = round((time.time() - start_retrieval) * 1000, 1)

    for cid, ctx in multi_ctx.card_contexts.items():
        name = multi_ctx.card_names.get(cid, cid)
        print(f"  • \033[92m{name}\033[0m: {len(ctx.parents)} parent sections, {len(ctx.citations)} citations")

    print(f"\n\033[93m[2/2] Streaming comparative synthesis (retrieval: {retrieval_ms}ms)...\033[0m\n")
    start_gen = time.time()
    token_count = 0
    for token in generator.generate_comparison_stream(args.query, multi_ctx):
        sys.stdout.write(token)
        sys.stdout.flush()
        token_count += 1

    gen_s = round(time.time() - start_gen, 2)
    print(f"\n\n\033[90m------------------------------------------------------------------------\033[0m")
    print(f"\033[90mComparison complete in {gen_s}s ({token_count} tokens streamed).\033[0m\n")

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify CLI with `--help` and dry invocation**

Run: `python scripts/compare_cards.py --help`  
Expected: Output showing arguments `--cards` and `--query`.

- [ ] **Step 3: Commit**

```bash
git add scripts/compare_cards.py
git commit -m "feat(cli): add standalone compare_cards terminal utility"
```

---

### Task 5: Frontend UI - Header Mode Switcher & Multi-Card Selector (`frontend/index.html`, `frontend/style.css`, `frontend/app.js`)

**Files:**
- Modify: `frontend/index.html`
- Modify: `frontend/style.css`
- Modify: `frontend/app.js`

**Interfaces:**
- Consumes: `/api/cards` response
- Produces: Interactive tab switcher, multi-card selection chips with 2–3 card constraints, updated comparative quick prompts.

- [ ] **Step 1: Update `frontend/index.html` with Mode Switcher & Multi-Card Selector**

Add view mode tabs in the header and multi-card selection controls:
```html
<!-- Inside .app-header between brand and controls: -->
<div class="view-mode-tabs" role="tablist">
    <button class="mode-tab active" id="singleCardModeBtn" role="tab" aria-selected="true">
        <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>
        <span>Single Card Chat</span>
    </button>
    <button class="mode-tab" id="compareModeBtn" role="tab" aria-selected="false">
        <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="7" height="18" rx="1"/><rect x="14" y="3" width="7" height="18" rx="1"/></svg>
        <span>Multi-Card Comparison</span>
    </button>
</div>

<!-- Inside .header-controls: -->
<div class="control-group single-card-control" id="singleCardGroup">
    <label for="cardSelect" class="control-label">Active Card</label>
    <div class="select-wrapper">
        <select id="cardSelect" aria-label="Select Credit Card">
            <option value="" disabled selected>Loading certificates...</option>
        </select>
    </div>
</div>

<div class="control-group multi-card-control" id="multiCardGroup" style="display: none;">
    <div class="multi-card-header">
        <label class="control-label">Compare Cards (<span id="selectedCardCount">0</span>/3)</label>
        <span class="selection-hint" id="selectionHint">Select 2 or 3 cards</span>
    </div>
    <div class="card-pills-selector" id="cardPillsSelector">
        <!-- Dynamically populated card chips with checkboxes -->
    </div>
</div>
```

- [ ] **Step 2: Add CSS rules in `frontend/style.css` for Mode Tabs and Multi-Card Selector**

Add styles for:
- `.view-mode-tabs`, `.mode-tab`, `.mode-tab.active`
- `.multi-card-control`, `.card-pills-selector`, `.card-pill-btn`, `.card-pill-btn.selected`
- `.selection-hint.warning` (subtle amber text when < 2 cards selected)

- [ ] **Step 3: Update `frontend/app.js` with Mode Switching & Card Multi-Selection State**

In `frontend/app.js`:
- Track `currentMode = "single"` or `"compare"`.
- Track `selectedCompareCards = new Set()`.
- Add tab click listeners to toggle UI between single card and multi-card mode.
- Render dynamic card pills in `cardPillsSelector` with toggle state.
- Update suggestion chips when in comparison mode with 5 comparison queries.

- [ ] **Step 4: Test in browser or via test suite to confirm zero console errors**

Run: `pytest tests/test_api.py::test_api_serves_frontend -v`  
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add frontend/index.html frontend/style.css frontend/app.js
git commit -m "feat(frontend): add comparison mode switcher and multi-card selector bar"
```

---

### Task 6: Frontend UI - Comparison Matrix Table Renderer & Segmented Citations (`frontend/app.js`, `frontend/style.css`)

**Files:**
- Modify: `frontend/app.js`
- Modify: `frontend/style.css`

**Interfaces:**
- Consumes: `/api/compare` SSE stream (`event: cards_context`, `event: token`, `event: citations`, `event: done`)
- Produces: Real-time Markdown table parsing, sleek dark fintech matrix table styling, segmented per-bank citation accordions.

- [ ] **Step 1: Add Markdown Table Parser & Formatter in `frontend/app.js`**

Implement `formatMarkdown(text)` enhanced to convert markdown tables (`| ... |`) into HTML tables wrapped in `.matrix-table-container`:
- Adds header `<th>` styling with column icons.
- Highlights the rightmost `Advantage / Winner` column with `.col-winner`.

- [ ] **Step 2: Implement `/api/compare` SSE Stream Consumer in `frontend/app.js`**

When in comparison mode, `handleSubmit`:
- Validates `selectedCompareCards.size >= 2 && selectedCompareCards.size <= 3`. If invalid, shows warning toast/hint.
- Sends POST request to `/api/compare` with `{card_ids: Array.from(selectedCompareCards), query}`.
- Handles `event: cards_context`: populates preliminary source badges.
- Handles `event: token`: live streams markdown into the assistant message body with table rendering.
- Handles `event: citations`: renders segmented accordion sections in the Sources sidebar (`TD`, `CIBC`, `Scotia`).

- [ ] **Step 3: Add CSS for Matrix Tables & Citations Accordion in `frontend/style.css`**

Add styles for:
- `.matrix-table-container`, `.matrix-table`, `.matrix-table th`, `.matrix-table td`
- `.matrix-table tr:hover`, `.matrix-table .col-winner`
- `.bank-accordion`, `.bank-accordion-header`, `.bank-accordion-content`

- [ ] **Step 4: Verify frontend responsiveness and table formatting**

Run: `pytest tests/test_api.py -v`  
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add frontend/app.js frontend/style.css
git commit -m "feat(frontend): add side-by-side matrix table rendering and segmented citation drawer"
```

---

### Task 7: Comprehensive 5-Rider Cross-Bank Ground Truth Benchmark (`tests/test_comparison.py`)

**Files:**
- Modify: `tests/test_comparison.py`

**Interfaces:**
- Consumes: TD Aeroplan VIP, CIBC Aeroplan VIP, Scotiabank Passport VIP indexes.
- Produces: Automated assertions verifying 100% precision across all 5 key insurance riders.

- [ ] **Step 1: Add Comprehensive 5-Rider Benchmark tests to `tests/test_comparison.py`**

Add fixture using local memory Qdrant / indexed cards and assert:
1. **Rental Car Collision Waiver**: TD ($85k / 48d) vs CIBC ($100k / 48d) vs Scotia ($65k / 31d).
2. **Emergency Travel Medical Duration (Age 65+)**: TD (4d) vs CIBC (10d) vs Scotia (3d).
3. **Trip Cancellation Reimbursement Limits**: Max trip dollar caps.
4. **Baggage Delay Thresholds**: 4 hours wait, $1,000 allowance.
5. **Mobile Device Protection**: $1,500 limit (TD/CIBC) vs $1,000 limit (Scotia).

- [ ] **Step 2: Run benchmark tests**

Run: `pytest tests/test_comparison.py -v`  
Expected: All comparison unit tests and 5-rider benchmarks PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/test_comparison.py
git commit -m "test(comparison): codify comprehensive 5-rider cross-bank comparison benchmark"
```

---

### Task 8: Documentation, Verification & Phase Status Update

**Files:**
- Create: `docs/phases/PHASE_6_COMPARISON.md`
- Modify: `docs/ROADMAP.md`
- Modify: `README.md`

**Interfaces:**
- Produces: Complete technical documentation of Phase 6 and roadmap status transition to **COMPLETE (100%) ✅**.

- [ ] **Step 1: Create `docs/phases/PHASE_6_COMPARISON.md`**

Document:
- Multi-Card Hybrid Retrieval Architecture & Fan-out design
- Comparative Synthesis Prompt & Grounding Rules
- API Reference for `POST /api/compare` with sample requests and SSE events
- Frontend Comparison Matrix UI & segmented citation drawer
- Benchmark verification table with ground truth figures for TD, CIBC, and Scotia

- [ ] **Step 2: Update `docs/ROADMAP.md` & `README.md`**

Update:
- Mark **Phase 6** as **COMPLETE (100%) ✅** with active test count and benchmark description.
- Add link to `phases/PHASE_6_COMPARISON.md`.
- Update Architecture overview to include Multi-Card Comparison Engine.

- [ ] **Step 3: Run full test suite across the entire project**

Run: `pytest tests/ -v`  
Expected: All tests PASS with zero regressions.

- [ ] **Step 4: Commit**

```bash
git add docs/phases/PHASE_6_COMPARISON.md docs/ROADMAP.md README.md
git commit -m "docs: complete Phase 6 documentation and mark roadmap phase complete"
```
