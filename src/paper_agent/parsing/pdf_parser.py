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


def parse_pdf(pdf_path: str) -> ParsedPaper:
    reader = PdfReader(pdf_path)
    pages_text = [page.extract_text() or "" for page in reader.pages]
    full_text = "\n".join(pages_text)

    # Split off reference section
    ref_section = ""
    m = _REF_SECTION_RE.search(full_text)
    if m:
        ref_section = full_text[m.end():]

    # Extract raw reference lines from ref section (lines starting with [N] or numbered)
    raw_refs: List[str] = []
    if ref_section:
        for line in ref_section.splitlines():
            line = line.strip()
            if line and re.match(r"^(\[\d+\]|\d+\.)", line):
                raw_refs.append(line)

    # Prefer extracting arXiv IDs from the reference section.
    arxiv_ids = list(dict.fromkeys(_ARXIV_RE.findall(ref_section))) if ref_section else []
    if not arxiv_ids:
        arxiv_ids = list(dict.fromkeys(_ARXIV_RE.findall(full_text)))

    return ParsedPaper(
        full_text=full_text,
        references_raw=raw_refs,
        arxiv_ids_found=arxiv_ids,
    )
