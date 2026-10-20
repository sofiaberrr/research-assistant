"""Tests for the FastAPI endpoints. The agent is replaced by a fake so the
tests are fast, free, and don't depend on Claude being reachable."""
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

pytestmark = pytest.mark.skipif(
    not (Path("data/papers.db").exists() and Path("chroma_db").exists()),
    reason="needs data/papers.db and chroma_db (run Steps 2 and 3 first)",
)


@pytest.fixture
def client(monkeypatch):
    import api.main

    def fake_ask(question):
        return {
            "question": question,
            "answer": "A surface code is ... [arXiv:2610.04613, p.1]",
            "tool_calls": [{"tool": "search_papers", "args": {"query": "surface code"}}],
            "context": "not returned by the API",
            "latency_s": 1.23,
        }

    monkeypatch.setattr(api.main, "ask", fake_ask)
    return TestClient(api.main.app)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_ask_returns_answer_and_tools(client):
    r = client.post("/ask", json={"question": "What is a surface code?"})
    assert r.status_code == 200
    body = r.json()
    assert "[arXiv:" in body["answer"]
    assert body["tool_calls"][0]["tool"] == "search_papers"
    assert "context" not in body  # internal field stays internal


def test_ask_rejects_empty_question(client):
    r = client.post("/ask", json={"question": ""})
    assert r.status_code == 422  # validation error, handled by Pydantic


def test_list_papers_respects_limit(client):
    r = client.get("/papers", params={"limit": 5})
    assert r.status_code == 200
    assert len(r.json()) == 5


def test_get_one_paper(client):
    paper_id = client.get("/papers", params={"limit": 1}).json()[0]["id"]
    r = client.get(f"/papers/{paper_id}")
    assert r.status_code == 200
    assert r.json()["id"] == paper_id


def test_unknown_paper_is_404(client):
    assert client.get("/papers/0000.00000").status_code == 404
