import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import pymupdf
import pymupdf4llm

@dataclass
class DocumentChunk:
    chunk_id: str
    card_name: str
    breadcrumb: str
    page_number: int
    content: str
    full_text_with_context: str

def normalize_text(text: str) -> str:
    """Normalizes Unicode hyphens, quotes, and whitespace into standard ASCII."""
    text = re.sub(r"[\u2010\u2011\u2012\u2013\u2014\u2015]", "-", text)
    text = re.sub(r"[\u2018\u2019\u201a\u201b]", "'", text)
    text = re.sub(r"[\u201c\u201d\u201e\u201f]", '"', text)
    text = re.sub(r"[\u00a0\u200b\u202f]", " ", text)
    text = re.sub(r"\ufffd", " ", text)
    return text

class InsuranceDocParser:
    """Parses digital insurance PDFs into hierarchical, breadcrumb-tagged Markdown chunks."""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def parse_pdf(self, pdf_path: str, card_name: str, card_id: Optional[str] = None,
                  extract_dir: Optional[str] = "data/extracted") -> List[DocumentChunk]:
        doc = pymupdf.open(pdf_path)
        toc = doc.get_toc()
        total_pages = len(doc)

        toc_policy_by_page, toc_section_by_page = self._build_toc_maps(toc, total_pages)

        pages_md: List[Tuple[int, str]] = []
        try:
            chunks_data = pymupdf4llm.to_markdown(doc, page_chunks=True)
            for item in chunks_data:
                meta = item.get("metadata", {})
                page_num = meta.get("page_number") or (meta.get("page", 0) + 1)
                raw_text = normalize_text(item.get("text", ""))
                pages_md.append((page_num, raw_text))
        except Exception:
            for page_idx in range(len(doc)):
                page_num = page_idx + 1
                raw_text = normalize_text(doc[page_idx].get_text("text"))
                pages_md.append((page_num, raw_text))
        finally:
            doc.close()

        if extract_dir:
            if not card_id:
                slug = re.sub(r"[^\w\s-]", "", Path(pdf_path).stem).strip().lower()
                card_id = re.sub(r"[-\s]+", "_", slug)
            self._save_extracted_markdown(pages_md, card_name, card_id, output_dir=extract_dir)

        return self._process_markdown_pages(pages_md, card_name, toc_policy_by_page, toc_section_by_page, toc)

    def _save_extracted_markdown(self, pages_md: List[Tuple[int, str]], card_name: str, 
                                card_id: str, output_dir: str = "data/extracted") -> Path:
        """Saves the complete converted Markdown document to data/extracted/<card_id>.md."""
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / f"{card_id}.md"

        content_parts = [
            f"# {card_name}",
            f"> Card ID: `{card_id}`  ",
            f"> Total Pages: {len(pages_md)}\n",
            "---\n"
        ]

        for page_num, text in pages_md:
            content_parts.append(f"<!-- Page {page_num} -->\n{text.strip()}\n")

        full_md = "\n".join(content_parts)
        out_file.write_text(full_md, encoding="utf-8")
        return out_file

    def _build_toc_maps(self, toc: List[Any], total_pages: int) -> Tuple[Dict[int, str], Dict[int, str]]:
        """Maps page numbers to policy names and section titles using the PDF outline (TOC)."""
        toc_policy_by_page: Dict[int, str] = {}
        toc_section_by_page: Dict[int, str] = {}

        if not toc:
            return toc_policy_by_page, toc_section_by_page

        # Find all Level 1 items (Policies)
        level1_items = [(normalize_text(t[1]).strip(), t[2]) for t in toc if t[0] == 1]
        if not level1_items:
            min_lvl = min(t[0] for t in toc)
            level1_items = [(normalize_text(t[1]).strip(), t[2]) for t in toc if t[0] == min_lvl]

        for idx, (pol_title, start_page) in enumerate(level1_items):
            end_page = level1_items[idx + 1][1] - 1 if idx + 1 < len(level1_items) else total_pages
            pol_sections = [
                (normalize_text(t[1]).strip(), t[2]) 
                for t in toc 
                if t[0] >= 2 and start_page <= t[2] <= end_page
            ]

            for p in range(start_page, end_page + 1):
                toc_policy_by_page[p] = pol_title
                sec_matches = [s for s in pol_sections if s[1] <= p]
                if sec_matches:
                    toc_section_by_page[p] = sec_matches[-1][0]
                else:
                    toc_section_by_page[p] = "Overview"

        return toc_policy_by_page, toc_section_by_page

    def _process_markdown_pages(self, pages_md: List[Tuple[int, str]], card_name: str,
                                toc_policy: Optional[Dict[int, str]] = None,
                                toc_section: Optional[Dict[int, str]] = None,
                                toc: Optional[List[Any]] = None) -> List[DocumentChunk]:
        chunks: List[DocumentChunk] = []
        current_policy = "General Policy Overview"
        current_section = "Introduction"
        chunk_counter = 0

        known_policies_map = {}
        known_sections = set()
        policy_starts = {}
        if toc:
            for item in toc:
                norm_title = normalize_text(item[1]).strip()
                if item[0] == 1:
                    known_policies_map[norm_title.lower()] = norm_title
                    if norm_title not in policy_starts:
                        policy_starts[norm_title] = item[2]
                else:
                    known_sections.add(norm_title.lower())

        for page_num, text in pages_md:
            # Check if this page contains an in-page policy header for a new policy
            has_in_page_policy_header = False
            if known_policies_map:
                for line in text.split("\n"):
                    clean = re.sub(r"[#*_|`]", "", re.sub(r"<[^>]+>", "", line)).strip().lower()
                    if clean in known_policies_map and known_policies_map[clean] != current_policy:
                        has_in_page_policy_header = True
                        break

            # Initialize policy or switch at page boundary if no in-page header will do it
            if current_policy == "General Policy Overview" and toc_policy and page_num in toc_policy:
                current_policy = toc_policy[page_num]
                current_section = toc_section.get(page_num, "Introduction")
            elif toc_policy and page_num in toc_policy and not has_in_page_policy_header:
                expected_policy = toc_policy[page_num]
                if expected_policy != current_policy and page_num >= policy_starts.get(expected_policy, page_num):
                    current_policy = expected_policy
                    current_section = toc_section.get(page_num, "Overview")

            lines = text.split("\n")
            buffer: List[str] = []

            for line in lines:
                stripped_line = line.strip()
                if not stripped_line:
                    buffer.append(line)
                    continue

                clean_text = re.sub(r"<[^>]+>", "", stripped_line).strip()
                clean_no_md = re.sub(r"[#*_|`]", "", clean_text).strip()

                h1_match = re.match(r"^#\s+(.+)$", stripped_line)
                h2_match = re.match(r"^##\s+(.+)$", stripped_line)
                h3_match = re.match(r"^###\s+(.+)$", stripped_line)
                sec_match = re.match(r"^(?:#+\s*)?(Section\s+\d+\s*[-–—:]\s*.+)$", clean_text, re.IGNORECASE)

                is_policy_header = False
                new_policy = None

                if known_policies_map and clean_no_md.lower() in known_policies_map:
                    is_policy_header = True
                    new_policy = known_policies_map[clean_no_md.lower()]
                elif not toc and h1_match:
                    h1_clean = re.sub(r"<[^>]+>", "", h1_match.group(1)).strip()
                    if not h1_clean.lower().startswith("section"):
                        is_policy_header = True
                        new_policy = h1_clean

                if is_policy_header and new_policy:
                    if buffer:
                        self._flush_buffer(buffer, card_name, current_policy, current_section, page_num, chunks, chunk_counter)
                        chunk_counter = len(chunks)
                        buffer = []
                    current_policy = new_policy
                    current_section = "Overview"
                else:
                    is_section_header = False
                    new_section = None

                    if sec_match:
                        is_section_header = True
                        new_section = sec_match.group(1).strip()
                    elif clean_text.lower() in known_sections:
                        is_section_header = True
                        new_section = clean_text
                    elif h1_match and not is_policy_header:
                        h1_clean = re.sub(r"<[^>]+>", "", h1_match.group(1)).strip()
                        is_section_header = True
                        new_section = h1_clean
                    elif h2_match or h3_match:
                        header_text = re.sub(r"<[^>]+>", "", (h2_match or h3_match).group(1)).strip()
                        if len(header_text) > 2:
                            is_section_header = True
                            new_section = header_text

                    if is_section_header and new_section:
                        if buffer:
                            self._flush_buffer(buffer, card_name, current_policy, current_section, page_num, chunks, chunk_counter)
                            chunk_counter = len(chunks)
                            buffer = []
                        current_section = new_section

                buffer.append(line)

            if buffer:
                self._flush_buffer(buffer, card_name, current_policy, current_section, page_num, chunks, chunk_counter)
                chunk_counter = len(chunks)

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
            step = max(self.chunk_size - self.chunk_overlap, 100)
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
