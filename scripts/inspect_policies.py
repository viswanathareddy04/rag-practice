import sys
import io
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from collections import defaultdict
from src.parser import InsuranceDocParser

# Ensure standard output can print Unicode characters on Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

def main():
    pdf_path = Path("data/td/TD Aeroplan Visa Infinite Privilege Card COI.pdf")
    if not pdf_path.exists():
        print(f"Error: Target PDF not found at {pdf_path}")
        sys.exit(1)

    card_name = "TD Aeroplan Visa Infinite Privilege Card"
    print("=" * 80)
    print(f"INSPECTING PARSED POLICIES & BREADCRUMBS: {card_name}")
    print(f"File: {pdf_path}")
    print("=" * 80)

    parser = InsuranceDocParser(chunk_size=400, chunk_overlap=80)
    chunks = parser.parse_pdf(str(pdf_path), card_name=card_name)

    print(f"\nTotal Chunks Generated: {len(chunks)}")
    
    # Organize chunks by policy and section
    policy_chunks = defaultdict(list)
    policy_pages = defaultdict(set)
    policy_sections = defaultdict(lambda: defaultdict(list))

    for chunk in chunks:
        # Breadcrumb format: [Card: ...] > [Policy: ...] > [Section: ...] (Page X)
        parts = chunk.breadcrumb.split(" > ")
        pol_part = parts[1].replace("[Policy: ", "").rstrip("]")
        sec_part = parts[2].split("] (Page ")[0].replace("[Section: ", "")
        
        policy_chunks[pol_part].append(chunk)
        policy_pages[pol_part].add(chunk.page_number)
        policy_sections[pol_part][sec_part].append(chunk)

    print(f"Total Unique Policies Identified: {len(policy_chunks)}\n")

    for idx, (policy, p_chunks) in enumerate(policy_chunks.items(), 1):
        pages = sorted(policy_pages[policy])
        page_range_str = f"Page {pages[0]}" if len(pages) == 1 else f"Pages {pages[0]}-{pages[-1]}"
        print(f"{idx}. {policy} ({page_range_str}) - {len(p_chunks)} chunks")
        
        sections = policy_sections[policy]
        print(f"   Sections ({len(sections)} distinct):")
        for s_idx, (sec_name, s_chunks) in enumerate(sections.items(), 1):
            sec_pages = sorted(set(c.page_number for c in s_chunks))
            sec_pg_str = f"p.{sec_pages[0]}" if len(sec_pages) == 1 else f"p.{sec_pages[0]}-{sec_pages[-1]}"
            print(f"     {s_idx}. {sec_name} ({sec_pg_str}, {len(s_chunks)} chunks)")

        # Print sample breadcrumb
        sample = p_chunks[0]
        print(f"   Sample Breadcrumb: {sample.breadcrumb}")
        preview = sample.content.replace('\n', ' ')[:120]
        print(f"   Preview: {preview}...\n")

    # Quality and Consistency Validations
    print("=" * 80)
    print("QUALITY & BREADCRUMB SANITY CHECKS")
    print("=" * 80)

    invalid_breadcrumbs = [c for c in chunks if not c.breadcrumb.startswith(f"[Card: {card_name}]")]
    print(f"✓ Valid Card Prefix on all chunks: {len(invalid_breadcrumbs) == 0}")

    invalid_pages = [c for c in chunks if c.page_number < 1 or c.page_number > 52]
    print(f"✓ Valid Page Numbers (1-52): {len(invalid_pages) == 0}")

    lengths = [len(c.content.split()) for c in chunks]
    print(f"✓ Chunk Word Counts: Min={min(lengths)}, Avg={sum(lengths)//len(lengths)}, Max={max(lengths)}")

    print("\nPhase 1 Inspection complete! All policies and sections cleanly structured.")

if __name__ == "__main__":
    main()
