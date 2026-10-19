"""
The 3 tools the agent can call.

The @tool decorator turns a normal Python function into something Claude can use.
Claude never sees the code, only the function NAME, its ARGUMENTS, and its DOCSTRING.
So the docstring is basically the instructions for when and how to use the tool.
"""
import sqlite3

import chromadb
from ddgs import DDGS
from langchain_core.tools import tool

DB_PATH = "data/papers.db"
CHROMA_DIR = "chroma_db"

# Open the Chroma collection once, when this file is first imported
_collection = chromadb.PersistentClient(path=CHROMA_DIR).get_collection("papers")


@tool
def search_papers(query: str, num_results: int = 8) -> str:
    """Search the full text of ~220 recent arXiv quantum computing papers.

    Use this for any question about what the papers say: methods, results,
    definitions, comparisons. Returns passages with their arXiv ID and page,
    which you must use to cite your sources.
    """
    results = _collection.query(query_texts=[query], n_results=num_results)
    passages = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        passages.append(
            f"[arXiv:{meta['paper_id']}, p.{meta['page']}] {meta['title']}\n{doc}"
        )
    return "\n\n---\n\n".join(passages) if passages else "No matching passages found."


@tool
def query_database(sql: str) -> str:
    """Run a read-only SQL (SQLite) query on the papers metadata database.

    Use this for counting, filtering, or listing papers by metadata, e.g.
    "how many papers are from 2026?" or "which papers have the most authors?".

    Table: papers
      id               TEXT     arXiv ID, e.g. '2610.04613'
      title            TEXT
      authors          TEXT     comma-separated names
      num_authors      INTEGER
      published_date   TEXT     'YYYY-MM-DD'
      year             INTEGER
      primary_category TEXT     e.g. 'quant-ph'
      summary          TEXT     the paper's abstract
    Only SELECT queries are allowed.
    """
    if not sql.strip().lower().startswith("select"):
        return "Error: only SELECT queries are allowed."
    try:
        # mode=ro opens the database read-only, so the agent can't modify anything
        conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
        cursor = conn.execute(sql)
        columns = [c[0] for c in cursor.description]
        rows = cursor.fetchmany(50)  # cap the output so it doesn't flood Claude
        conn.close()
    except Exception as e:
        return f"SQL error: {e}"  # Claude can read this and fix its query
    if not rows:
        return "Query returned no rows."
    lines = [" | ".join(columns)] + [" | ".join(str(v) for v in row) for row in rows]
    return "\n".join(lines)


@tool
def web_search(query: str) -> str:
    """Search the web. Use this only for information that is NOT in the paper
    collection: recent news, background concepts, companies, people, or
    anything after the papers were published.
    """
    try:
        results = DDGS().text(query, max_results=5)
    except Exception as e:
        return f"Web search failed: {e}"
    return "\n\n".join(
        f"{r['title']} ({r['href']})\n{r['body']}" for r in results
    ) or "No web results."


ALL_TOOLS = [search_papers, query_database, web_search]
