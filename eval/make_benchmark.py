"""
Step 5a: Build the 40-question benchmark (eval/benchmark.json).

Run from the project root (only needs to be run ONCE):
    python -m eval.make_benchmark

- 30 "paper" questions: Claude writes a question + reference answer from a
  random passage in the index. We know which paper it came from, so later we
  can check whether the agent cites the right source.
- 10 "database" questions: reference answers are computed directly with SQL,
  so they're guaranteed correct.

Open benchmark.json afterwards and skim it. Delete or fix any weird questions,
a benchmark is only as good as its questions.
"""
import json
import random
import sqlite3
from pathlib import Path

import chromadb
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

DB_PATH = "data/papers.db"
OUT_PATH = Path("eval/benchmark.json")
NUM_PAPER_QUESTIONS = 30
GEN_MODEL = "claude-haiku-5-5"  # cheaper model is plenty for writing questions
random.seed(42)  # same "random" picks every run, so the benchmark is reproducible

client = Anthropic(max_retries=5)


def reply_text(reply) -> str:
    """Get the text from Claude's reply, skipping any 'thinking' blocks."""
    return "".join(b.text for b in reply.content if b.type == "text")


def parse_json(text: str) -> dict:
    """Pull the JSON object out of Claude's reply."""
    return json.loads(text[text.find("{") : text.rfind("}") + 1])


def paper_questions() -> list[dict]:
    collection = chromadb.PersistentClient(path="chroma_db").get_collection("papers")
    everything = collection.get(include=["documents", "metadatas"])

    # Group passages by paper, then pick one decent-length passage from 30 different papers
    by_paper: dict[str, list] = {}
    for doc, meta in zip(everything["documents"], everything["metadatas"]):
        if len(doc) >= 700:
            by_paper.setdefault(meta["paper_id"], []).append((doc, meta))
    chosen = random.sample(sorted(by_paper), NUM_PAPER_QUESTIONS)

    questions = []
    for i, paper_id in enumerate(chosen, start=1):
        doc, meta = random.choice(by_paper[paper_id])
        prompt = f"""Here is a passage from the paper "{meta['title']}":

<passage>
{doc}
</passage>

Write ONE question that this passage clearly answers, plus the correct answer.
- The question must make sense on its own to someone who hasn't seen the passage.
  Never say "this paper" or "the passage"; name the specific method, result, or concept instead.
- Prefer questions about a concrete result, method, or definition.
- The answer should be 1-2 sentences, based only on the passage.

Reply with only JSON: {{"question": "...", "answer": "..."}}"""
        reply = client.messages.create(
            model=GEN_MODEL, max_tokens=2000, messages=[{"role": "user", "content": prompt}]
        )
        qa = parse_json(reply_text(reply))
        questions.append({
            "id": f"paper_{i:02d}",
            "type": "paper",
            "question": qa["question"],
            "reference_answer": qa["answer"],
            "source_paper_id": paper_id,
        })
        print(f"[{i}/{NUM_PAPER_QUESTIONS}] {qa['question'][:90]}")
    return questions


def database_questions() -> list[dict]:
    conn = sqlite3.connect(DB_PATH)
    one = lambda sql: conn.execute(sql).fetchone()[0]  # run a query, get the single value

    busiest_month = conn.execute(
        "SELECT substr(published_date, 1, 7) AS m, COUNT(*) FROM papers "
        "GROUP BY m ORDER BY COUNT(*) DESC LIMIT 1"
    ).fetchone()

    pairs = [
        ("How many papers are in the collection?",
         one("SELECT COUNT(*) FROM papers")),
        (f"How many papers in the collection were published in {busiest_month[0]} (year-month)?",
         busiest_month[1]),
        ("What is the largest number of authors on a single paper in the collection?",
         one("SELECT MAX(num_authors) FROM papers")),
        ("What is the average number of authors per paper, rounded to one decimal?",
         one("SELECT ROUND(AVG(num_authors), 1) FROM papers")),
        ("How many papers in the collection have exactly one author?",
         one("SELECT COUNT(*) FROM papers WHERE num_authors = 1")),
        ("How many papers have 10 or more authors?",
         one("SELECT COUNT(*) FROM papers WHERE num_authors >= 10")),
        ("What is the earliest publication date of any paper in the collection?",
         one("SELECT MIN(published_date) FROM papers")),
        ("What is the most recent publication date of any paper in the collection?",
         one("SELECT MAX(published_date) FROM papers")),
        ("How many papers have the word 'error' in their title?",
         one("SELECT COUNT(*) FROM papers WHERE LOWER(title) LIKE '%error%'")),
        ("How many papers have the word 'algorithm' in their title?",
         one("SELECT COUNT(*) FROM papers WHERE LOWER(title) LIKE '%algorithm%'")),
    ]
    conn.close()
    return [
        {"id": f"db_{i:02d}", "type": "database", "question": q,
         "reference_answer": str(a), "source_paper_id": None}
        for i, (q, a) in enumerate(pairs, start=1)
    ]


if __name__ == "__main__":
    benchmark = paper_questions() + database_questions()
    OUT_PATH.write_text(json.dumps(benchmark, indent=2, ensure_ascii=False))
    print(f"\nSaved {len(benchmark)} questions to {OUT_PATH}")
