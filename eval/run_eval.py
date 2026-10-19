"""
Step 5b: Run the agent on the benchmark and score it.

Run from the project root:
    python -m eval.run_eval baseline

The word after the command is a label for this run (e.g. "baseline", "new_prompt").
Results are saved to eval/results/<label>.json so you can compare runs.

Metrics:
- Accuracy:         does the answer match the reference? (LLM-as-judge)
- Groundedness:     is every claim supported by what the tools returned? (LLM-as-judge)
- Latency:          seconds per question (average, median, 95th percentile)
- Source hit rate:  for paper questions, did the answer cite the paper the question came from?
"""
import json
import statistics
import sys
from pathlib import Path

from anthropic import Anthropic
from dotenv import load_dotenv

from agent.graph import ask

load_dotenv()

BENCHMARK = Path("eval/benchmark.json")
RESULTS_DIR = Path("eval/results")
JUDGE_MODEL = "claude-haiku-5-5"

judge_client = Anthropic(max_retries=5)


def reply_text(reply) -> str:
    """Get the text from Claude's reply, skipping any 'thinking' blocks."""
    return "".join(b.text for b in reply.content if b.type == "text")

JUDGE_PROMPT = """You are grading an AI research assistant.

<question>{question}</question>
<reference_answer>{reference}</reference_answer>
<assistant_answer>{answer}</assistant_answer>
<retrieved_context>
{context}
</retrieved_context>

Grade two things:
1. accuracy: 1 if the assistant's answer contains the key facts of the reference
   answer (wording can differ, extra correct detail is fine), else 0.
2. grounded: 1 if every factual claim in the assistant's answer is supported by
   the retrieved context, else 0. Saying "I couldn't find this" counts as grounded.

Reply with only JSON:
{{"accuracy": 0 or 1, "grounded": 0 or 1, "reason": "one short sentence"}}"""


def judge(item: dict, result: dict) -> dict:
    prompt = JUDGE_PROMPT.format(
        question=item["question"],
        reference=item["reference_answer"],
        answer=result["answer"],
        # v1 cut this at 15,000 chars, which hid real evidence from the judge on
        # multi-search answers. 80,000 chars fits everything the agent saw.
        context=result["context"][:80000],
    )
    # The judge occasionally returns slightly broken JSON (e.g. stray quotes),
    # so try up to 3 times before giving up.
    for attempt in range(3):
        reply = judge_client.messages.create(
            model=JUDGE_MODEL, max_tokens=2000, messages=[{"role": "user", "content": prompt}]
        )
        text = reply_text(reply)
        try:
            return json.loads(text[text.find("{") : text.rfind("}") + 1])
        except json.JSONDecodeError:
            if attempt == 2:
                raise


def pct(values: list) -> str:
    return f"{100 * sum(values) / len(values):.0f}%" if values else "n/a"


def main() -> None:
    label = sys.argv[1] if len(sys.argv) > 1 else "run"
    benchmark = json.loads(BENCHMARK.read_text())
    rows = []

    for i, item in enumerate(benchmark, start=1):
        try:
            result = ask(item["question"])
            grade = judge(item, result)
        except Exception as e:  # one failure shouldn't kill the whole run
            print(f"[{i}/{len(benchmark)}] ERROR: {e}")
            rows.append({**item, "error": str(e)})
            continue

        cited = None
        if item["source_paper_id"]:
            cited = int(item["source_paper_id"] in result["answer"])

        row = {
            **item,
            "answer": result["answer"],
            "tools_used": [c["tool"] for c in result["tool_calls"]],
            "latency_s": result["latency_s"],
            "accuracy": grade["accuracy"],
            "grounded": grade["grounded"],
            "cited_source": cited,
            "judge_reason": grade["reason"],
            "context": result["context"],  # saved so answers can be re-judged later
        }
        rows.append(row)
        mark = "✓" if row["accuracy"] else "✗"
        print(f"[{i}/{len(benchmark)}] {mark} acc={row['accuracy']} grounded={row['grounded']} "
              f"{row['latency_s']:5.1f}s  {item['question'][:60]}")

    ok = [r for r in rows if "error" not in r]
    latencies = sorted(r["latency_s"] for r in ok)
    paper_rows = [r for r in ok if r["type"] == "paper"]
    db_rows = [r for r in ok if r["type"] == "database"]

    summary = {
        "label": label,
        "questions": len(benchmark),
        "errors": len(rows) - len(ok),
        "accuracy": pct([r["accuracy"] for r in ok]),
        "accuracy_paper_questions": pct([r["accuracy"] for r in paper_rows]),
        "accuracy_database_questions": pct([r["accuracy"] for r in db_rows]),
        "groundedness": pct([r["grounded"] for r in ok]),
        "source_hit_rate": pct([r["cited_source"] for r in paper_rows]),
        "latency_avg_s": round(statistics.mean(latencies), 2) if latencies else None,
        "latency_median_s": round(statistics.median(latencies), 2) if latencies else None,
        "latency_p95_s": round(latencies[int(0.95 * (len(latencies) - 1))], 2) if latencies else None,
    }

    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / f"{label}.json"
    out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2, ensure_ascii=False))

    print("\n===== SUMMARY =====")
    for k, v in summary.items():
        print(f"{k:30s} {v}")
    print(f"\nFull results saved to {out}")

    # If a baseline run exists, show a before/after comparison
    baseline_file = RESULTS_DIR / "baseline.json"
    if label != "baseline" and baseline_file.exists():
        before = json.loads(baseline_file.read_text())["summary"]
        print(f"\n===== {label} vs baseline =====")
        for k in ["accuracy", "groundedness", "source_hit_rate", "latency_avg_s", "latency_p95_s"]:
            print(f"{k:30s} {str(before[k]):>8s}  ->  {summary[k]}")


if __name__ == "__main__":
    main()
