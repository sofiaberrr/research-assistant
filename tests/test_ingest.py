"""Tests for the chunking and bibliography filter (pure functions, no data needed)."""
from ingest.build_index import chunk_text, clean_text, is_reference_chunk, CHUNK_SIZE


def test_clean_text_removes_junk_and_extra_spaces():
    assert clean_text("hello\x00   world\n\n") == "hello world"


def test_chunks_are_capped_at_chunk_size():
    chunks = chunk_text("a" * 5000)
    assert len(chunks) > 1
    assert all(len(c) <= CHUNK_SIZE for c in chunks)


def test_chunks_overlap():
    text = "".join(str(i % 10) for i in range(3000))
    first, second = chunk_text(text)[:2]
    assert first[-200:] == second[:200]  # last 200 chars of one = first 200 of the next


def test_tiny_text_produces_no_chunks():
    assert chunk_text("too short") == []


def test_bibliography_is_flagged():
    refs = ("[1] A. B. Smith, Phys. Rev. A 54, 1098 (1996). [2] C. D. Jones et al., "
            "arXiv:2401.01234 (2024). [3] E. F. Lee, Phys. Rev. Lett. 77, 793 (1996). "
            "[4] G. H. Wang, https://doi.org/10.1103/x (2012).")
    assert is_reference_chunk(refs)


def test_normal_paragraph_is_kept():
    body = ("Quantum error correction protects logical information by encoding it "
            "redundantly across many physical qubits. The surface code has a threshold "
            "near 1% under circuit-level noise [12], making it a leading candidate.")
    assert not is_reference_chunk(body)
