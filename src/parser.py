import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple
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

class InsuranceDocParser:
    """Parses digital insurance PDFs into hierarchical, breadcrumb-tagged Markdown chunks."""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def parse_pdf(self, pdf_path: str, card_name: str) -> List[DocumentChunk]:
        pages_md: List[Tuple[int, str]] = []
        doc = pymupdf.open(pdf_path)
        for page_idx in range(len(doc)):
            page_num = page_idx + 1
            # pymupdf4llm extracts markdown preserving tables and headers
            try:
                md_text = pymupdf4llm.to_markdown(doc, pages=[page_idx])
            except Exception:
                # Fallback to plain text if markdown layout extraction errors
                md_text = doc[page_idx].get_text("text")
            pages_md.append((page_num, md_text))
        doc.close()

        return self._process_markdown_pages(pages_md, card_name)

    def _process_markdown_pages(self, pages_md: List[Tuple[int, str]], card_name: str) -> List[DocumentChunk]:
        chunks: List[DocumentChunk] = []
        current_policy = "General Policy Overview"
        current_section = "Introduction"
        chunk_counter = 0

        for page_num, text in pages_md:
            lines = text.split("\n")
            buffer: List[str] = []

            for line in lines:
                h1_match = re.match(r"^#\s+(.+)$", line)
                h2_match = re.match(r"^##\s+(.+)$", line)
                h3_match = re.match(r"^###\s+(.+)$", line)

                if h1_match:
                    if buffer:
                        self._flush_buffer(buffer, card_name, current_policy, current_section, page_num, chunks, chunk_counter)
                        chunk_counter = len(chunks)
                        buffer = []
                    current_policy = h1_match.group(1).strip()
                    current_section = "Overview"
                elif h2_match or h3_match:
                    if buffer:
                        self._flush_buffer(buffer, card_name, current_policy, current_section, page_num, chunks, chunk_counter)
                        chunk_counter = len(chunks)
                        buffer = []
                    header_text = (h2_match or h3_match).group(1).strip()
                    current_section = header_text

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
