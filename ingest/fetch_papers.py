"""
Step 2: Download papers from arXiv and save their metadata to a SQLite database.

Run from the project root:
    python -m ingest.fetch_papers

What it does:
1. Searches arXiv for papers matching TOPIC_QUERY
2. Downloads each paper's PDF into data/pdfs/
3. Saves metadata (title, authors, year, category...) into data/papers.db

The database is what your SQL tool will query later.
The PDFs are what your RAG pipeline will read in Step 3.
"""
import sqlite3
import time
import urllib.request
from pathlib import Path

import arxiv

# ---- Settings you can change ----
TOPIC_QUERY = "cat:quant-ph AND abs:\"quantum computing\""  # arXiv search syntax
NUM_PAPERS = 220  # a bit over 200 in case a few downloads fail
# ---------------------------------

DATA_DIR = Path("data")
PDF_DIR = DATA_DIR / "pdfs"
DB_PATH = DATA_DIR / "papers.db"


def setup_database(conn: sqlite3.Connection) -> None:
    """Create the papers table if it doesn't exist yet."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS papers (
            id               TEXT PRIMARY KEY,   -- arXiv ID, e.g. 2401.01234
            title            TEXT NOT NULL,
            authors          TEXT NOT NULL,      -- comma-separated names
            num_authors      INTEGER NOT NULL,
            published_date   TEXT NOT NULL,      -- YYYY-MM-DD
            year             INTEGER NOT NULL,
            primary_category TEXT,
            summary          TEXT,
            pdf_path         TEXT
        )
        """
    )
    conn.commit()


def download_pdf(url: str, dest: Path) -> bool:
    """Download one PDF. Returns True if it worked."""
    if dest.exists():
        return True  # already downloaded on a previous run
    try:
        urllib.request.urlretrieve(url, dest)
        return True
    except Exception as e:
        print(f"   ! download failed: {e}")
        return False


def main() -> None:
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    setup_database(conn)

    # arXiv asks for ~3 seconds between API calls, the client handles that
    client = arxiv.Client(page_size=100, delay_seconds=3, num_retries=5)
    search = arxiv.Search(
        query=TOPIC_QUERY,
        max_results=NUM_PAPERS,
        sort_by=arxiv.SortCriterion.SubmittedDate,
    )

    saved = 0
    for i, paper in enumerate(client.results(search), start=1):
        paper_id = paper.get_short_id().split("v")[0]  # drop version suffix
        pdf_path = PDF_DIR / f"{paper_id}.pdf"
        print(f"[{i}/{NUM_PAPERS}] {paper.title[:70]}")

        if not download_pdf(paper.pdf_url, pdf_path):
            continue

        authors = [a.name for a in paper.authors]
        conn.execute(
            "INSERT OR REPLACE INTO papers VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                paper_id,
                paper.title.strip(),
                ", ".join(authors),
                len(authors),
                paper.published.strftime("%Y-%m-%d"),
                paper.published.year,
                paper.primary_category,
                paper.summary.strip(),
                str(pdf_path),
            ),
        )
        conn.commit()
        saved += 1
        time.sleep(1)  # be polite to arXiv's servers

    conn.close()
    print(f"\nDone! Saved {saved} papers to {DB_PATH} and PDFs to {PDF_DIR}/")


if __name__ == "__main__":
    main()
