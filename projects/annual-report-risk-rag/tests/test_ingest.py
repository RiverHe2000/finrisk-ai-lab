from __future__ import annotations

from pathlib import Path

import pytest

from report_rag.ingest import chunk_document, iter_sections, load_directory, load_document
from report_rag.schemas import Document


def test_iter_sections_splits_on_headings() -> None:
    text = "intro line\n# A\nbody a\n## B\nbody b1\nbody b2\n"
    assert list(iter_sections(text)) == [
        ("preamble", "intro line"),
        ("A", "body a"),
        ("B", "body b1\nbody b2"),
    ]


def test_chunks_never_cross_sections(tiny_doc: Document) -> None:
    chunks = chunk_document(tiny_doc, chunk_size=5, chunk_overlap=1)
    sections = {c.section for c in chunks}
    assert sections == {"Capital", "Liquidity"}
    assert [c.position for c in chunks] == list(range(len(chunks)))
    assert all(c.chunk_id == f"tiny:{c.position:04d}" for c in chunks)


def test_chunk_overlap_repeats_words(tiny_doc: Document) -> None:
    chunks = [c for c in chunk_document(tiny_doc, 6, 2) if c.section == "Capital"]
    assert len(chunks) >= 2
    first, second = chunks[0].text.split(), chunks[1].text.split()
    assert first[-2:] == second[:2]


def test_chunk_size_must_exceed_overlap(tiny_doc: Document) -> None:
    with pytest.raises(ValueError, match="must exceed"):
        chunk_document(tiny_doc, chunk_size=10, chunk_overlap=10)


def test_load_document_infers_title_and_id(tmp_path: Path) -> None:
    p = tmp_path / "some_bank_2025.md"
    p.write_text("# Some Bank Annual Report\n\ntext", encoding="utf-8")
    doc = load_document(p)
    assert doc.doc_id == "some_bank_2025"
    assert doc.title == "Some Bank Annual Report"
    assert doc.source_path == str(p)


def test_load_document_rejects_unknown_suffix(tmp_path: Path) -> None:
    p = tmp_path / "x.docx"
    p.write_bytes(b"")
    with pytest.raises(ValueError, match="Unsupported"):
        load_document(p)


def test_load_directory_is_sorted_and_filtered(tmp_path: Path) -> None:
    (tmp_path / "b.md").write_text("# B", encoding="utf-8")
    (tmp_path / "a.txt").write_text("plain", encoding="utf-8")
    (tmp_path / "ignore.json").write_text("{}", encoding="utf-8")
    docs = load_directory(tmp_path)
    assert [d.doc_id for d in docs] == ["a", "b"]
    assert docs[0].title == "A"
