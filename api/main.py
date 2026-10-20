"""
Step 6: Serve the agent as a REST API with FastAPI.

Run from the project root:
    uvicorn api.main:app --reload

Then open http://localhost:8000/docs for an interactive page where you can
try every endpoint in your browser.

4 endpoints:
    GET  /health              is the server up?
    POST /ask                 ask the agent a question
    GET  /papers              list papers (optional filter by year)
    GET  /papers/{paper_id}   details for one paper
"""
import sqlite3

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from agent.graph import ask

DB_PATH = "data/papers.db"

app = FastAPI(
    title="Agentic Research Assistant",
    description="Ask questions about ~220 arXiv quantum computing papers. "
    "Built with Claude, LangGraph, and Chroma.",
)


# ---------- Request / response shapes ----------
# Pydantic models describe exactly what goes in and out. FastAPI uses them to
# validate requests automatically and to generate the /docs page.

class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000,
                          examples=["What is a surface code and why is it popular?"])


class ToolCall(BaseModel):
    tool: str
    args: dict


class AskResponse(BaseModel):
    question: str
    answer: str
    tool_calls: list[ToolCall]
    latency_s: float


class Paper(BaseModel):
    id: str
    title: str
    authors: str
    year: int
    published_date: str


class PaperDetail(Paper):
    num_authors: int
    primary_category: str | None
    summary: str | None


# ---------- Helpers ----------

def query_db(sql: str, params: tuple = ()) -> list[dict]:
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row  # lets us turn rows into dicts
    rows = [dict(r) for r in conn.execute(sql, params).fetchall()]
    conn.close()
    return rows


# ---------- Endpoints ----------

@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask_question(request: AskRequest) -> dict:
    try:
        result = ask(request.question)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Agent failed: {e}")
    return result  # extra fields (like the raw context) are dropped by response_model


@app.get("/papers", response_model=list[Paper])
def list_papers(
    year: int | None = Query(None, description="Only papers from this year"),
    limit: int = Query(20, ge=1, le=250),
) -> list[dict]:
    sql = "SELECT id, title, authors, year, published_date FROM papers"
    params: tuple = ()
    if year is not None:
        sql += " WHERE year = ?"  # the ? placeholder prevents SQL injection
        params = (year,)
    sql += " ORDER BY published_date DESC LIMIT ?"
    return query_db(sql, params + (limit,))


@app.get("/papers/{paper_id}", response_model=PaperDetail)
def get_paper(paper_id: str) -> dict:
    rows = query_db(
        "SELECT id, title, authors, num_authors, year, published_date, "
        "primary_category, summary FROM papers WHERE id = ?",
        (paper_id,),
    )
    if not rows:
        raise HTTPException(status_code=404, detail=f"Paper {paper_id} not found")
    return rows[0]
