"""Document loading and section-aware chunking."""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator
from pathlib import Path

from report_rag.schemas import Chunk, Document

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")
_WORD_RE = re.compile(r"\S+")

SUPPORTED_SUFFIXES = {".md", ".txt", ".pdf"}


def load_document(path: Path, doc_id: str | None = None) -> Document:
    """Load a Markdown/plain-text/PDF file into a :class:`Document`.

    PDF support requires the optional ``pdf`` extra (``pypdf``).
    """
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        msg = f"Unsupported file type '{suffix}' for {path}"
        raise ValueError(msg)
    text = _read_pdf(path) if suffix == ".pdf" else path.read_text(encoding="utf-8")
    title = _first_heading(text) or path.stem.replace("_", " ").title()
    return Document(doc_id=doc_id or path.stem, title=title, text=text, source_path=str(path))


def load_directory(directory: Path) -> list[Document]:
    """Load every supported file in ``directory`` (sorted for determinism)."""
    paths = sorted(p for p in directory.iterdir() if p.suffix.lower() in SUPPORTED_SUFFIXES)
    return [load_document(p) for p in paths]


def _read_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        msg = "PDF ingestion needs the 'pdf' extra: uv sync --extra pdf"
        raise ImportError(msg) from exc
    reader = PdfReader(str(path))
    return "\n\n".join(page.extract_text() or "" for page in reader.pages)


def _first_heading(text: str) -> str | None:
    for line in text.splitlines():
        match = _HEADING_RE.match(line)
        if match:
            return match.group(2)
    return None


def iter_sections(text: str) -> Iterator[tuple[str, str]]:
    """Yield ``(section_title, body)`` pairs split on Markdown headings.

    Text before the first heading is attributed to the section ``"preamble"``.
    """
    current_title = "preamble"
    buffer: list[str] = []
    for line in text.splitlines():
        match = _HEADING_RE.match(line)
        if match:
            if body := "\n".join(buffer).strip():
                yield current_title, body
            current_title = match.group(2)
            buffer = []
        else:
            buffer.append(line)
    if body := "\n".join(buffer).strip():
        yield current_title, body


def _windows(words: list[str], size: int, overlap: int) -> Iterable[list[str]]:
    if size <= overlap:
        msg = f"chunk_size ({size}) must exceed chunk_overlap ({overlap})"
        raise ValueError(msg)
    step = size - overlap
    start = 0
    while start < len(words):
        yield words[start : start + size]
        if start + size >= len(words):
            break
        start += step


def chunk_document(doc: Document, chunk_size: int = 220, chunk_overlap: int = 40) -> list[Chunk]:
    """Split a document into overlapping word windows that never cross a section boundary.

    Keeping chunks within a section means the section title travels with each
    chunk as metadata, which both improves retrieval (the title is prepended to
    the indexed text) and gives the LLM the table/section context it needs.
    """
    chunks: list[Chunk] = []
    position = 0
    for section, body in iter_sections(doc.text):
        words = _WORD_RE.findall(body)
        for window in _windows(words, chunk_size, chunk_overlap):
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.doc_id}:{position:04d}",
                    doc_id=doc.doc_id,
                    section=section,
                    text=" ".join(window),
                    position=position,
                )
            )
            position += 1
    return chunks
