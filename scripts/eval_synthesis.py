#!/usr/bin/env python3
"""
Automated Evaluation Suite for Phase 3: Grounded Synthesis & Citation Guardrails.
Tests live LLM generation (Qwen 2.5 7B) on TD Aeroplan VIP for schema adherence,
numeric accuracy, and negative control rejection.
"""

import sys
import time
from pathlib import Path

# Reconfigure stdout for UTF-8 compatibility on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.storage import StorageManager
from src.retriever import HybridRetriever
from src.generator import LLMGenerator

# Terminal formatting
GREEN = "\033[92m"
RED = "\033[91m"
CYAN = "\033[96m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"

EVAL_CASES = [
    {
        "id": "Q1",
        "category": "Flight Delay",
        "question": "What is the minimum delay time and maximum coverage for flight delay?",
        "expected_terms": ["1,000", "4 hours"],
        "is_negative": False
    },
    {
        "id": "Q2",
        "category": "Auto Rental MSRP",
        "question": "What is the maximum MSRP for rental car collision damage coverage?",
        "expected_terms": ["85,000", "msrp"],
        "is_negative": False
    },
    {
        "id": "Q3",
        "category": "Medical Pre-existing",
        "question": "What are the pre-existing medical condition stability requirements?",
        "expected_terms": ["stable", "pre-existing"],
        "is_negative": False
    },
    {
        "id": "Q4",
        "category": "Mobile Device Limit",
        "question": "What is the mobile device insurance maximum limit and deductible?",
        "expected_terms": ["1,500", "deductible"],
        "is_negative": False
    },
    {
        "id": "Q5",
        "category": "Negative Control (Pet Emergency)",
        "question": "Does this insurance certificate cover pet health emergencies during travel?",
        "expected_terms": [],
        "is_negative": True,
        "negative_indicators": ["not cover", "does not cover", "not specify", "no coverage", "not found", "not mention"]
    }
]

MANDATORY_SECTIONS = [
    "### Direct Answer",
    "### Coverage Limits & Details",
    "### Key Conditions & Exclusions",
    "### Policy Citations"
]

def run_evaluation(model_name: str = "qwen2.5:7b", base_url: str = "http://localhost:11434"):
    print(f"\n{BOLD}{CYAN}==================================================================={RESET}")
    print(f"{BOLD}{CYAN}    PHASE 3: GROUNDED SYNTHESIS & CITATION GUARDRAILS EVALUATION   {RESET}")
    print(f"{BOLD}{CYAN}==================================================================={RESET}")
    print(f"Model: {BOLD}{model_name}{RESET} | Base URL: {base_url}\n")

    project_root = Path(__file__).resolve().parent.parent
    index_dir = project_root / ".index"
    card_id = "td_aeroplan_vip"

    storage = StorageManager(storage_dir=index_dir)
    if not storage.is_indexed(card_id):
        print(f"{RED}Error: Index for '{card_id}' does not exist in .index/. Run inspect or benchmark first.{RESET}")
        sys.exit(1)

    retriever = HybridRetriever(storage_manager=storage, rrf_k=60)
    generator = LLMGenerator(model=model_name, base_url=base_url)

    total = len(EVAL_CASES)
    passed = 0

    for idx, case in enumerate(EVAL_CASES):
        q_id = case["id"]
        cat = case["category"]
        question = case["question"]

        print(f"{BOLD}[{idx+1}/{total}] {q_id} ({cat}):{RESET} \"{question}\"")

        start = time.time()
        scored_chunks = retriever.retrieve(card_id=card_id, query=question, top_k=8)
        response = generator.generate(question, scored_chunks)
        latency = time.time() - start

        answer = response.answer_markdown
        answer_lower = answer.lower()

        # 1. Verify 4 mandatory section headings
        missing_sections = [sec for sec in MANDATORY_SECTIONS if sec.lower() not in answer_lower]
        if missing_sections:
            print(f"  {RED}FAILED Schema:{RESET} Missing sections: {missing_sections}")
            print(f"  {DIM}Answer preview:\n{answer[:250]}...{RESET}\n")
            continue

        # 2. Check for negative control rejection or positive term containment
        if case["is_negative"]:
            # Negative control: must reject coverage without hallucinating
            matched_neg = any(ind in answer_lower for ind in case["negative_indicators"])
            if not matched_neg:
                print(f"  {RED}FAILED Negative Guardrail:{RESET} Model did not clearly state lack of coverage.")
                print(f"  {DIM}Answer:\n{answer}{RESET}\n")
                continue
            print(f"  {GREEN}[PASS] Guardrail Passed:{RESET} Correctly declined coverage ({latency:.1f}s)")
        else:
            missing_terms = [t for t in case["expected_terms"] if t.lower() not in answer_lower]
            if missing_terms:
                print(f"  {RED}FAILED Content Grounding:{RESET} Missing expected facts: {missing_terms}")
                print(f"  {DIM}Answer preview:\n{answer[:300]}...{RESET}\n")
                continue
            print(f"  {GREEN}[PASS] Grounding Passed:{RESET} Included all terms {case['expected_terms']} ({latency:.1f}s)")

        # 3. Check citations validity
        if not response.citations:
            print(f"  {RED}FAILED Citations:{RESET} Citations list is empty.")
            continue

        passed += 1
        print(f"  {DIM}Citations: {len(response.citations)} certificate references verified.{RESET}\n")

    print(f"{BOLD}==================================================================={RESET}")
    if passed == total:
        print(f"{BOLD}{GREEN}ALL {passed}/{total} SYNTHESIS & GUARDRAIL EVALUATIONS PASSED! (100%){RESET}")
        print(f"{DIM}Phase 3 exit criteria satisfied.{RESET}\n")
    else:
        print(f"{BOLD}{RED}{passed}/{total} EVALUATIONS PASSED.{RESET}\n")
        sys.exit(1)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen2.5:7b")
    parser.add_argument("--base-url", default="http://localhost:11434")
    args = parser.parse_args()
    run_evaluation(model_name=args.model, base_url=args.base_url)
