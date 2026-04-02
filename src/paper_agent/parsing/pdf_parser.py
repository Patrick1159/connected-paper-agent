from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List

from pypdf import PdfReader


@dataclass
class ParsedPaper:
    full_text: str
    references_raw: List[str] = field(default_factory=list)  # raw reference strings
    arxiv_ids_found: List[str] = field(default_factory=list)  # arXiv IDs extracted


_ARXIV_RE = re.compile(r"(?:arXiv:|arxiv\.org/(?:abs|pdf)/)([\d]{4}\.[\d]{4,5}(?:v\d+)?)", re.IGNORECASE)
_REF_SECTION_RE = re.compile(r"\n(?:References|Bibliography|REFERENCES|BIBLIOGRAPHY)\s*\n", re.IGNORECASE)
_REF_ITEM_START_RE = re.compile(r"^(\[\d+\]|\d+\.)\s*")


def _extract_reference_blocks(ref_section: str) -> List[str]:
    blocks: List[str] = []
    current_lines: List[str] = []

    for raw_line in ref_section.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        if _REF_ITEM_START_RE.match(line):
            if current_lines:
                blocks.append(" ".join(current_lines))
            current_lines = [line]
        elif current_lines:
            current_lines.append(line)

    if current_lines:
        blocks.append(" ".join(current_lines))

    return blocks


def parse_pdf(pdf_path: str) -> ParsedPaper:
    reader = PdfReader(pdf_path)
    pages_text = [page.extract_text() or "" for page in reader.pages]
    full_text = "\n".join(pages_text)

    # Split off reference section
    ref_section = ""
    m = _REF_SECTION_RE.search(full_text)
    if m:
        ref_section = full_text[m.end():]

    # Extract raw reference blocks from the reference section.
    raw_refs: List[str] = _extract_reference_blocks(ref_section) if ref_section else []

    # Prefer extracting arXiv IDs from the reference section.
    arxiv_ids = list(dict.fromkeys(_ARXIV_RE.findall(ref_section))) if ref_section else []
    if not arxiv_ids:
        arxiv_ids = list(dict.fromkeys(_ARXIV_RE.findall(full_text)))

    return ParsedPaper(
        full_text=full_text,
        references_raw=raw_refs,
        arxiv_ids_found=arxiv_ids,
    )
