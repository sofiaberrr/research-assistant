"""Tests for the agent's tools. These use your real database and index
(they're skipped if you haven't built them yet), but never call Claude."""
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    not (Path("data/papers.db").exists() and Path("chroma_db").exists()),
    reason="needs data/papers.db and chroma_db (run Steps 2 and 3 first)",
)


def test_sql_tool_counts_papers():
    from agent.tools import query_database
    out = query_database.invoke({"sql": "SELECT COUNT(*) AS n FROM papers"})
    count = int(out.splitlines()[1])
    assert count >= 200


def test_sql_tool_blocks_writes():
    from agent.tools import query_database
    out = query_database.invoke({"sql": "DROP TABLE papers"})
    assert "only SELECT" in out


def test_sql_tool_returns_errors_instead_of_crashing():
    from agent.tools import query_database
    out = query_database.invoke({"sql": "SELECT nope FROM papers"})
    assert out.startswith("SQL error")


def test_search_returns_citable_passages():
    from agent.tools import search_papers
    out = search_papers.invoke({"query": "quantum error correction", "num_results": 3})
    assert out.count("[arXiv:") == 3  # every passage carries its paper ID for citing
    assert ", p." in out              # ...and its page number
