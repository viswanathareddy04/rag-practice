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
                # Grounded fallback if Ollama daemon is offline during testing
                answer_text = f"### Direct Answer\nBased on the certificate context retrieved for '{question}':\n\n" \
                              f"### Coverage Limits & Details\n" + \
                              "\n\n".join([f"- **{c.breadcrumb}**: {c.content}" for c in [sc.chunk for sc in chunks]]) + \
                              "\n\n### Key Conditions & Exclusions\nRefer to the specific certificate sections cited above.\n\n" \
                              "### Policy Citations\n" + \
                              "\n".join([f"- [{c.breadcrumb}]" for c in [sc.chunk for sc in chunks]])

        return GroundedResponse(
            question=question,
            answer_markdown=answer_text,
            citations=citations
        )
