"""
Step 3: Turn the PDFs into searchable chunks in Chroma (the RAG index).

Run from the project root:
    python -m ingest.build_index

What it does:
1. Reads the list of papers from data/papers.db
2. Extracts the text from each PDF, page by page
3. Cuts the text into overlapping chunks (~1000 characters each)
4. Stores each chunk in Chroma, which turns it into an embedding automatically
   (the first run downloads a small embedding model, ~80MB)

Every chunk keeps its paper ID, title, and page number, which is how
the agent will cite its sources later.
"""
import re
import sqlite3
from pathlib import Path

import chromadb
from pypdf import PdfReader

DB_PATH = Path("data/papers.db")
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "papers"

CHUNK_SIZE = 1000    # characters per chunk (roughly a paragraph or two)
CHUNK_OVERLAP = 200  # chunks overlap so an idea split across a boundary isn't lost
FILTER_REFERENCES = True  # drop bibliography chunks (set False to compare in Step 5)

# Patterns that show up constantly in reference lists but rarely in normal text
CITATION_PATTERNS = re.compile(
    r"\[\d{1,3}\]"                    # [12]
    r"|Phys\.\s?Rev\."                # Phys. Rev.
    r"|arXiv:\s?\d{4}\.\d{4,5}"       # arXiv:2401.01234
    r"|\(\d{4}\)"                     # (2019)
    r"|\bet al\."                     # et al.
    r"|\bdoi\b|https?://"             # links and DOIs
    r"|\b[A-Z]\.\s?[A-Z]?\.?\s?[A-Z][a-z]+,"  # initials + surname, e.g. "P. W. Shor,"
)


def is_reference_chunk(chunk: str) -> bool:
    """A chunk with lots of citation patterns is almost certainly a bibliography."""
    return len(CITATION_PATTERNS.findall(chunk)) >= 8


def clean_text(text: str) -> str:
    """Remove junk characters and squash extra whitespace."""
    text = text.replace("\x00", "")
    return re.sub(r"\s+", " ", text).strip()


def chunk_text(text: str) -> list[str]:
    """Split text into overlapping windows of CHUNK_SIZE characters."""
    chunks = []
    start = 0
    while start < len(text):
        chunk = text[start : start + CHUNK_SIZE]
        if len(chunk) > 100:  # skip tiny leftover scraps
            chunks.append(chunk)
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


def main() -> None:
    conn = sqlite3.connect(DB_PATH)
    papers = conn.execute("SELECT id, title, year, pdf_path FROM papers").fetchall()
    conn.close()

    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_or_create_collection(COLLECTION_NAME)

    total_chunks = 0
    skipped = 0
    for i, (paper_id, title, year, pdf_path) in enumerate(papers, start=1):
        try:
            reader = PdfReader(pdf_path)
        except Exception as e:
            print(f"[{i}/{len(papers)}] ! couldn't open {paper_id}: {e}")
            continue

        ids, docs, metas = [], [], []
        for page_num, page in enumerate(reader.pages, start=1):
            try:
                text = clean_text(page.extract_text() or "")
            except Exception:
                continue  # some pages (images, weird encodings) can't be read
            for c, chunk in enumerate(chunk_text(text)):
                if FILTER_REFERENCES and is_reference_chunk(chunk):
                    skipped += 1
                    continue
                ids.append(f"{paper_id}_p{page_num}_c{c}")
                docs.append(chunk)
                metas.append({"paper_id": paper_id, "title": title, "year": year, "page": page_num})

        if docs:
            # upsert = insert, or overwrite if it already exists, so reruns are safe
            collection.upsert(ids=ids, documents=docs, metadatas=metas)
        total_chunks += len(docs)
        print(f"[{i}/{len(papers)}] {len(docs):4d} chunks  {title[:60]}")

    print(f"\nDone! {total_chunks} chunks from {len(papers)} papers stored in {CHROMA_DIR}/")
    print(f"Skipped {skipped} bibliography chunks")


if __name__ == "__main__":
    main()
