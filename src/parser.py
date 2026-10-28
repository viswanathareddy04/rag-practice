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

@dataclass
class ParentSection:
    parent_id: str          # e.g., "cibc_aeroplan_vip__travel_medical__sec8"
    card_id: str            # e.g., "cibc_aeroplan_vip"
    card_name: str          # e.g., "CIBC Aeroplan VIP"
    policy_name: str        # e.g., "TRAVEL MEDICAL INSURANCE"
    section_title: str      # e.g., "SECTION 8 - WHAT ARE YOU NOT COVERED FOR?"
    page_start: int         # Starting PDF page (1-indexed)
    page_end: int           # Ending PDF page (1-indexed)
    content: str            # Full Markdown text of the section (~600–1,500 words)

@dataclass
class ChildChunk:
    child_id: str           # e.g., "cibc_aeroplan_vip__travel_medical__sec8__c001"
    parent_id: str          # Foreign key referencing ParentSection.parent_id
    card_id: str
    card_name: str
    policy_name: str
    section_title: str
    page_number: int        # Exact PDF page where this child clause appears
    content: str            # Specific clause/paragraph text (~100–150 words)
    breadcrumb: str         # "[Card: Name] > [Policy] > [Section] (Page X)"
    parent_content: str     # Denormalized copy of the full parent section text

def normalize_text(text: str) -> str:
    """Normalizes Unicode hyphens, quotes, and whitespace into standard ASCII."""
    text = re.sub(r"[\u2010\u2011\u2012\u2013\u2014\u2015]", "-", text)
    text = re.sub(r"[\u2018\u2019\u201a\u201b]", "'", text)
    text = re.sub(r"[\u201c\u201d\u201e\u201f]", '"', text)
    text = re.sub(r"[\u00a0\u200b\u202f]", " ", text)
    text = re.sub(r"\ufffd", " ", text)
    return text

def clean_heading(text: str) -> str:
    """Strips HTML tags (<mark>, <u>, etc.), markdown symbols, and extra whitespace."""
    clean = re.sub(r"<[^>]+>", "", text)
    clean = re.sub(r"[#*_|`]", "", clean)
    return clean.strip()

def slugify(text: str) -> str:
    """Converts a heading into a clean, safe identifier slug."""
    clean = clean_heading(text).lower()
    clean = re.sub(r"[^\w\s-]", "", clean)
    slug = re.sub(r"[-\s]+", "_", clean).strip("_")
    return slug or "section"

class InsuranceDocParser:
    """Parses digital insurance PDFs into hierarchical ParentSections and ChildChunks."""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def parse_markdown_text(self, markdown_text: str, card_name: str, card_id: str) -> Tuple[List[ParentSection], List[ChildChunk]]:
        """Universally parses standardized Markdown into ParentSections and ChildChunks using heading hierarchy."""
        lines = markdown_text.split("\n")
        parents: List[ParentSection] = []
        children: List[ChildChunk] = []

        current_page = 1
        current_policy = "General Policy Overview"
        current_section = "Overview"
        section_page_start = 1
        section_lines: List[str] = []
        section_page_map: Dict[int, int] = {}  # maps line_idx within section to page_num

        def flush_current_section():
            nonlocal section_lines, section_page_start, current_policy, current_section
            raw_content = "\n".join(section_lines).strip()
            if not raw_content or len(raw_content) < 20:
                section_lines = []
                section_page_map.clear()
                return

            page_end = current_page
            pol_slug = slugify(current_policy)
            sec_slug = slugify(current_section)
            parent_id = f"{card_id}__{pol_slug}__{sec_slug}"
            
            # Ensure unique parent_id if duplicate section titles appear
            existing_count = sum(1 for p in parents if p.parent_id.startswith(parent_id))
            if existing_count > 0:
                parent_id = f"{parent_id}_{existing_count + 1}"

            parent = ParentSection(
                parent_id=parent_id,
                card_id=card_id,
                card_name=card_name,
                policy_name=current_policy,
                section_title=current_section,
                page_start=section_page_start,
                page_end=page_end,
                content=raw_content
            )
            parents.append(parent)

            # Generate Child Chunks from this section
            child_chunks = self._chunk_parent_section(parent, section_lines, section_page_map, section_page_start)
            children.extend(child_chunks)

            section_lines = []
            section_page_map.clear()

        line_idx_in_sec = 0
        for line in lines:
            # Check for page comment
            page_match = re.match(r"<!--\s*Page\s+(\d+)\s*-->", line, re.IGNORECASE)
            if page_match:
                current_page = int(page_match.group(1))
                continue

            stripped = line.strip()
            if not stripped:
                section_lines.append(line)
                section_page_map[line_idx_in_sec] = current_page
                line_idx_in_sec += 1
                continue

            h1_match = re.match(r"^#\s+(.+)$", stripped)
            h2_match = re.match(r"^##\s+(.+)$", stripped)
            h3_match = re.match(r"^###\s+(.+)$", stripped)
            h4_match = re.match(r"^####\s+(.+)$", stripped)
            sec_match = re.match(r"^(?:#+\s*)?(?:<[^>]+>\s*)?(?:Section|SECTION)\s+\d+\s*[-–—:]\s*(.+)$", stripped)

            # Level 1 / Level 2 Heading: Policy transition
            if h1_match or (h2_match and not sec_match):
                heading_raw = (h1_match or h2_match).group(1)
                heading_clean = clean_heading(heading_raw)
                is_boilerplate = heading_clean.lower() in [
                    "certificate of insurance", "certificates of insurance", 
                    "certificate", "certificates", "table of contents"
                ]
                # Check if it looks like a certificate title or policy name
                if len(heading_clean) > 3 and not heading_clean.lower().startswith("section") and not is_boilerplate:
                    flush_current_section()
                    current_policy = heading_clean
                    current_section = "Overview"
                    section_page_start = current_page
                    line_idx_in_sec = 0

            # Level 3 / Level 4 Heading or explicit Section Header: Section transition
            elif h3_match or h4_match or sec_match:
                if sec_match:
                    sec_clean = clean_heading(stripped)
                else:
                    sec_clean = clean_heading((h3_match or h4_match).group(1))

                if len(sec_clean) > 2:
                    flush_current_section()
                    current_section = sec_clean
                    section_page_start = current_page
                    line_idx_in_sec = 0

            section_lines.append(line)
            section_page_map[line_idx_in_sec] = current_page
            line_idx_in_sec += 1

        flush_current_section()
        return parents, children

    def _chunk_parent_section(self, parent: ParentSection, lines: List[str], 
                              page_map: Dict[int, int], default_page: int) -> List[ChildChunk]:
        """Splits a ParentSection into granular ChildChunks (~80-150 words), preserving table integrity."""
        children: List[ChildChunk] = []
        raw_text = parent.content

        # 1. Detect if entire section is a table block
        is_table = any(line.strip().startswith("|") for line in lines)
        if is_table:
            # Table-aware chunking: preserve table rows
            table_lines = [l for l in lines if l.strip().startswith("|")]
            header_lines = table_lines[:2] if len(table_lines) >= 2 and "---" in table_lines[1] else []
            data_rows = table_lines[2:] if header_lines else table_lines

            # If small table, keep as single child
            total_words = len(" ".join(table_lines).split())
            if total_words <= 150 or len(data_rows) <= 3:
                chunk_id = f"{parent.parent_id}__c{len(children)+1:03d}"
                page_num = page_map.get(0, default_page)
                breadcrumb = f"[Card: {parent.card_name}] > [{parent.policy_name}] > [{parent.section_title}] (Page {page_num})"
                children.append(ChildChunk(
                    child_id=chunk_id,
                    parent_id=parent.parent_id,
                    card_id=parent.card_id,
                    card_name=parent.card_name,
                    policy_name=parent.policy_name,
                    section_title=parent.section_title,
                    page_number=page_num,
                    content="\n".join(table_lines),
                    breadcrumb=breadcrumb,
                    parent_content=parent.content
                ))
                return children
            else:
                # Group rows in clusters of 2-3
                cluster_size = 3
                for i in range(0, len(data_rows), cluster_size):
                    chunk_rows = header_lines + data_rows[i:i + cluster_size]
                    chunk_text = "\n".join(chunk_rows)
                    chunk_id = f"{parent.parent_id}__c{len(children)+1:03d}"
                    page_num = page_map.get(i, default_page)
                    breadcrumb = f"[Card: {parent.card_name}] > [{parent.policy_name}] > [{parent.section_title}] (Page {page_num})"
                    children.append(ChildChunk(
                        child_id=chunk_id,
                        parent_id=parent.parent_id,
                        card_id=parent.card_id,
                        card_name=parent.card_name,
                        policy_name=parent.policy_name,
                        section_title=parent.section_title,
                        page_number=page_num,
                        content=chunk_text,
                        breadcrumb=breadcrumb,
                        parent_content=parent.content
                    ))
                return children

        # 2. Text-aware chunking: split by paragraphs (\n\n) and bullet clusters
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", raw_text) if p.strip()]
        current_chunk_words: List[str] = []
        current_chunk_page = default_page

        for para in paragraphs:
            para_words = para.split()
            if not para_words:
                continue

            # If paragraph itself is within ~150 words, accumulate or flush
            if len(para_words) <= 150:
                if len(current_chunk_words) + len(para_words) <= 160:
                    current_chunk_words.extend(para_words)
                else:
                    if current_chunk_words:
                        chunk_text = " ".join(current_chunk_words)
                        chunk_id = f"{parent.parent_id}__c{len(children)+1:03d}"
                        breadcrumb = f"[Card: {parent.card_name}] > [{parent.policy_name}] > [{parent.section_title}] (Page {current_chunk_page})"
                        children.append(ChildChunk(
                            child_id=chunk_id,
                            parent_id=parent.parent_id,
                            card_id=parent.card_id,
                            card_name=parent.card_name,
                            policy_name=parent.policy_name,
                            section_title=parent.section_title,
                            page_number=current_chunk_page,
                            content=chunk_text,
                            breadcrumb=breadcrumb,
                            parent_content=parent.content
                        ))
                    current_chunk_words = list(para_words)
            else:
                # Flush previous accumulation
                if current_chunk_words:
                    chunk_text = " ".join(current_chunk_words)
                    chunk_id = f"{parent.parent_id}__c{len(children)+1:03d}"
                    breadcrumb = f"[Card: {parent.card_name}] > [{parent.policy_name}] > [{parent.section_title}] (Page {current_chunk_page})"
                    children.append(ChildChunk(
                        child_id=chunk_id,
                        parent_id=parent.parent_id,
                        card_id=parent.card_id,
                        card_name=parent.card_name,
                        policy_name=parent.policy_name,
                        section_title=parent.section_title,
                        page_number=current_chunk_page,
                        content=chunk_text,
                        breadcrumb=breadcrumb,
                        parent_content=parent.content
                    ))
                    current_chunk_words = []

                # Split long paragraph by sentence boundaries
                sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", para) if s.strip()]
                sent_accum: List[str] = []
                for s in sentences:
                    s_words = s.split()
                    if len(sent_accum) + len(s_words) <= 150:
                        sent_accum.extend(s_words)
                    else:
                        if sent_accum:
                            chunk_text = " ".join(sent_accum)
                            chunk_id = f"{parent.parent_id}__c{len(children)+1:03d}"
                            breadcrumb = f"[Card: {parent.card_name}] > [{parent.policy_name}] > [{parent.section_title}] (Page {current_chunk_page})"
                            children.append(ChildChunk(
                                child_id=chunk_id,
                                parent_id=parent.parent_id,
                                card_id=parent.card_id,
                                card_name=parent.card_name,
                                policy_name=parent.policy_name,
                                section_title=parent.section_title,
                                page_number=current_chunk_page,
                                content=chunk_text,
                                breadcrumb=breadcrumb,
                                parent_content=parent.content
                            ))
                        sent_accum = list(s_words)
                if sent_accum:
                    current_chunk_words = list(sent_accum)

        if current_chunk_words:
            chunk_text = " ".join(current_chunk_words)
            chunk_id = f"{parent.parent_id}__c{len(children)+1:03d}"
            breadcrumb = f"[Card: {parent.card_name}] > [{parent.policy_name}] > [{parent.section_title}] (Page {current_chunk_page})"
            children.append(ChildChunk(
                child_id=chunk_id,
                parent_id=parent.parent_id,
                card_id=parent.card_id,
                card_name=parent.card_name,
                policy_name=parent.policy_name,
                section_title=parent.section_title,
                page_number=current_chunk_page,
                content=chunk_text,
                breadcrumb=breadcrumb,
                parent_content=parent.content
            ))

        return children

    def parse_pdf(self, pdf_path: str, card_name: str, card_id: Optional[str] = None,
                  extract_dir: Optional[str] = "data/extracted") -> List[DocumentChunk]:
        """Legacy-compatible method: parses digital insurance PDFs and returns DocumentChunk list."""
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

    def parse_pdf_parent_child(self, pdf_path: str, card_name: str, card_id: Optional[str] = None,
                              extract_dir: Optional[str] = "data/extracted") -> Tuple[List[ParentSection], List[ChildChunk]]:
        """Parses a digital insurance PDF into ParentSections and ChildChunks using universal Markdown structure."""
        doc = pymupdf.open(pdf_path)
        pages_md: List[Tuple[int, str]] = []
        try:
            chunks_data = pymupdf4llm.to_markdown(doc, page_chunks=True)
            for item in chunks_data:
                meta = item.get("metadata", {})
                page_num = meta.get("page_number") or (meta.get("page", 0) + 1)
                raw_text = normalize_text(item.get("text", ""))
                pages_md.append((page_num, raw_text))
        finally:
            doc.close()

        if not card_id:
            slug = re.sub(r"[^\w\s-]", "", Path(pdf_path).stem).strip().lower()
            card_id = re.sub(r"[-\s]+", "_", slug)

        full_md_parts = []
        for page_num, text in pages_md:
            full_md_parts.append(f"<!-- Page {page_num} -->\n{text.strip()}")

        full_md = "\n\n".join(full_md_parts)
        if extract_dir:
            self._save_extracted_markdown(pages_md, card_name, card_id, output_dir=extract_dir)

        return self.parse_markdown_text(full_md, card_name=card_name, card_id=card_id)

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
            has_in_page_policy_header = False
            if known_policies_map:
                for line in text.split("\n"):
                    clean = re.sub(r"[#*_|`]", "", re.sub(r"<[^>]+>", "", line)).strip().lower()
                    if clean in known_policies_map and known_policies_map[clean] != current_policy:
                        has_in_page_policy_header = True
                        break

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
