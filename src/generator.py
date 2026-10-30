import json
import os
import re
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Generator, Union
import httpx

from src.retriever import ScoredChunk, RetrievedContext, MultiCardContext

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

    def _build_context_prompt(self, question: str, context_or_chunks: Union[RetrievedContext, List[ScoredChunk]]) -> str:
        context_blocks = []

        if isinstance(context_or_chunks, RetrievedContext):
            for i, p in enumerate(context_or_chunks.parents):
                page_str = f"Page {p.page_start}" if p.page_start == p.page_end else f"Pages {p.page_start}-{p.page_end}"
                context_blocks.append(
                    f"--- Parent Section {i+1}: [{p.policy_name}] > {p.section_title} ({page_str}) ---\n"
                    f"{p.content}\n"
                )
        else:
            for i, sc in enumerate(context_or_chunks):
                c = sc.chunk
                context_blocks.append(f"--- Context Block {i+1} ---\n{c.breadcrumb}\n{c.content}\n")
        
        context_text = "\n".join(context_blocks)
        return f"CONTEXT INFORMATION:\n{context_text}\n\nUSER QUESTION:\n{question}\n\nAnswer according to the rules and structure:"

    def _format_fallback_response(self, question: str, context_or_chunks: Union[RetrievedContext, List[ScoredChunk]]) -> str:
        if isinstance(context_or_chunks, RetrievedContext):
            detail_lines = []
            citation_lines = []
            for p in context_or_chunks.parents:
                page_str = f"Page {p.page_start}" if p.page_start == p.page_end else f"Pages {p.page_start}-{p.page_end}"
                detail_lines.append(f"- **[{p.policy_name}] {p.section_title} ({page_str})**: {p.content[:300]}...")
                citation_lines.append(f"- [{p.section_title}, {page_str}]")

            return f"### Direct Answer\nBased on the certificate context retrieved for '{question}':\n\n" \
                   f"### Coverage Limits & Details\n" + \
                   "\n\n".join(detail_lines) + \
                   "\n\n### Key Conditions & Exclusions\nRefer to the specific certificate sections cited above.\n\n" \
                   "### Policy Citations\n" + \
                   "\n".join(citation_lines)
        else:
            return f"### Direct Answer\nBased on the certificate context retrieved for '{question}':\n\n" \
                   f"### Coverage Limits & Details\n" + \
                   "\n\n".join([f"- **{c.breadcrumb}**: {c.content}" for c in [sc.chunk for sc in context_or_chunks]]) + \
                   "\n\n### Key Conditions & Exclusions\nRefer to the specific certificate sections cited above.\n\n" \
                   "### Policy Citations\n" + \
                   "\n".join([f"- [{c.breadcrumb}]" for c in [sc.chunk for sc in context_or_chunks]])

    def extract_citations(self, context_or_chunks: Union[RetrievedContext, List[ScoredChunk]]) -> List[Dict[str, Any]]:
        if isinstance(context_or_chunks, RetrievedContext):
            return context_or_chunks.citations

        citations = []
        for sc in context_or_chunks:
            c = sc.chunk
            citations.append({
                "page_number": c.page_number,
                "breadcrumb": c.breadcrumb,
                "snippet": c.content[:200] + "..." if len(c.content) > 200 else c.content,
                "score": round(sc.score, 4)
            })
        return citations

    def generate_stream(self, question: str, context_or_chunks: Union[RetrievedContext, List[ScoredChunk]]) -> Generator[str, None, None]:
        """Yields streaming tokens as they are generated by the model."""
        user_prompt = self._build_context_prompt(question, context_or_chunks)

        if self.ollama_client:
            res = self.ollama_client.chat(model=self.model, messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ], stream=True)
            for part in res:
                if isinstance(part, dict) and "message" in part and "content" in part["message"]:
                    yield part["message"]["content"]
            return

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            "stream": True
        }

        client_timeout = httpx.Timeout(120.0, connect=30.0)
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
            fallback_text = self._format_fallback_response(question, context_or_chunks)
            yield fallback_text

    def generate(self, question: str, context_or_chunks: Union[RetrievedContext, List[ScoredChunk]]) -> GroundedResponse:
        user_prompt = self._build_context_prompt(question, context_or_chunks)
        citations = self.extract_citations(context_or_chunks)

        if self.ollama_client:
            res = self.ollama_client.chat(model=self.model, messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ])
            answer_text = res["message"]["content"]
        else:
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                "stream": False
            }
            client_timeout = httpx.Timeout(120.0, connect=30.0)
            try:
                response = httpx.post(f"{self.base_url}/api/chat", json=payload, timeout=client_timeout)
                response.raise_for_status()
                data = response.json()
                answer_text = data["message"]["content"]
            except Exception:
                answer_text = self._format_fallback_response(question, context_or_chunks)

        return GroundedResponse(
            question=question,
            answer_markdown=answer_text,
            citations=citations
        )

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

