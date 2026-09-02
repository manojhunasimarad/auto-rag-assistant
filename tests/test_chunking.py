from src.chunking import build_chunks, chunk_text


def test_chunk_text_respects_size_roughly():
    text = "\n\n".join(f"Paragraph number {i}. " * 20 for i in range(10))
    chunks = chunk_text(text, chunk_size=200, overlap=40)

    assert len(chunks) > 1
    # A single paragraph can legitimately be forced to exceed chunk_size when
    # even one sentence alone is longer than chunk_size (the packer never
    # drops content), but no chunk should wildly blow past the budget.
    assert all(len(c) <= 200 * 2 for c in chunks)


def test_chunk_text_overlap_preserves_boundary_context():
    overlap = 50
    text = "\n\n".join([f"Sentence {i} of the document." for i in range(30)])
    chunks = chunk_text(text, chunk_size=150, overlap=overlap)

    assert len(chunks) >= 2
    # chunk[i+1] is built as tail(chunk[i], overlap) + "\n\n" + next_unit, so
    # it must start with exactly the last `overlap` characters of chunk[i] --
    # that's the entire point of overlap: no boundary content is lost.
    for i in range(len(chunks) - 1):
        expected_prefix = chunks[i][-overlap:]
        assert chunks[i + 1].startswith(expected_prefix)


def test_chunk_text_no_empty_chunks():
    text = "Just one short paragraph."
    chunks = chunk_text(text, chunk_size=800, overlap=120)
    assert chunks == ["Just one short paragraph."]


def test_chunk_text_rejects_bad_overlap():
    import pytest

    with pytest.raises(ValueError):
        chunk_text("hello", chunk_size=100, overlap=100)  # overlap must be < chunk_size


def test_build_chunks_tags_source_and_index(tmp_path):
    (tmp_path / "a.md").write_text("Alpha paragraph one.\n\nAlpha paragraph two.", encoding="utf-8")
    (tmp_path / "b.md").write_text("Beta paragraph one.", encoding="utf-8")

    chunks = build_chunks(tmp_path, chunk_size=1000, overlap=100)

    sources = {c.source for c in chunks}
    assert sources == {"a.md", "b.md"}
    # each source's chunks are indexed starting at 0
    a_chunks = [c for c in chunks if c.source == "a.md"]
    assert [c.chunk_index for c in a_chunks] == list(range(len(a_chunks)))
