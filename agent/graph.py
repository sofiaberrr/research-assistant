"""
Step 4: The agent itself, built as a LangGraph graph.

Run from the project root:
    python -m agent.graph "your question here"

The graph has 2 nodes and loops between them:

    START -> agent --(Claude wants a tool)--> tools --> agent --> ...
                   |--(Claude is done)------> END

- "agent" node: sends the conversation to Claude, which either answers
  or asks to call one or more tools.
- "tools" node: actually runs the tools Claude asked for and adds the
  results to the conversation.
"""
import sys
import time

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, MessagesState, StateGraph

from agent.tools import ALL_TOOLS

load_dotenv()

MODEL = "claude-sonnet-5-5"

SYSTEM_PROMPT = """You are a research assistant for a collection of ~220 recent arXiv \
papers on quantum computing.

Tools:
- search_papers: what the papers SAY (methods, results, explanations)
- query_database: facts ABOUT the papers (counts, dates, authors, titles)
- web_search: only for things not covered by the collection

Rules:
- Only state facts that appear in the tool results. Do not add background knowledge,
  derivations, equation numbers, or extra details from memory, even if you're confident
  they're true.
- Cite every claim from the papers like [arXiv:2610.04613, p.3], using only paper IDs
  and page numbers that actually appear in the tool results.
- If a search doesn't contain the answer, search again before giving up: rephrase with
  more specific terms (method names, quantities, notation) or different keywords.
  Try up to 3 searches.
- If the results still only partly answer the question, answer that part and say
  what's missing. Never fill the gap yourself.
- Answer directly in a few sentences, but always include the key specifics from the
  passages (numbers, thresholds, scalings, names). No tangents."""

llm = ChatAnthropic(model=MODEL, max_tokens=2048).bind_tools(ALL_TOOLS)
TOOLS_BY_NAME = {t.name: t for t in ALL_TOOLS}


# ---------- Nodes ----------

def agent_node(state: MessagesState) -> dict:
    """Ask Claude what to do next, given the conversation so far."""
    response = llm.invoke([SystemMessage(SYSTEM_PROMPT)] + state["messages"])
    return {"messages": [response]}


def tools_node(state: MessagesState) -> dict:
    """Run every tool Claude asked for and return the results."""
    last_message = state["messages"][-1]
    results = []
    for call in last_message.tool_calls:
        tool = TOOLS_BY_NAME[call["name"]]
        output = tool.invoke(call["args"])
        results.append(ToolMessage(content=str(output), tool_call_id=call["id"], name=call["name"]))
    return {"messages": results}


def should_continue(state: MessagesState) -> str:
    """After Claude responds: did it ask for tools, or is it done?"""
    return "tools" if state["messages"][-1].tool_calls else END


# ---------- Build the graph ----------

builder = StateGraph(MessagesState)
builder.add_node("agent", agent_node)
builder.add_node("tools", tools_node)
builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", should_continue, ["tools", END])
builder.add_edge("tools", "agent")
graph = builder.compile()


# ---------- Helper used by the CLI, the eval (Step 5) and the API (Step 6) ----------

def _text_of(message) -> str:
    """Claude's reply can be a plain string or a list of blocks; get just the text."""
    if isinstance(message.content, str):
        return message.content
    return "".join(b.get("text", "") for b in message.content if isinstance(b, dict))


def ask(question: str) -> dict:
    """Run the agent on one question. Returns the answer, tools used, and latency."""
    start = time.perf_counter()
    final_state = graph.invoke(
        {"messages": [HumanMessage(question)]},
        {"recursion_limit": 20},  # safety net so it can't loop forever
    )
    latency = time.perf_counter() - start

    tool_calls = []
    context_parts = []  # every tool call AND its result, in order
    for msg in final_state["messages"]:
        for call in getattr(msg, "tool_calls", None) or []:
            tool_calls.append({"tool": call["name"], "args": call["args"]})
            context_parts.append(f"TOOL CALL: {call['name']}({call['args']})")
        if isinstance(msg, ToolMessage):
            context_parts.append(f"TOOL RESULT ({msg.name}):\n{msg.content}")

    return {
        "question": question,
        "answer": _text_of(final_state["messages"][-1]),
        "tool_calls": tool_calls,
        "context": "\n\n".join(context_parts),  # what the agent saw; used for groundedness in Step 5
        "latency_s": round(latency, 2),
    }


if __name__ == "__main__":
    q = sys.argv[1] if len(sys.argv) > 1 else "What is a surface code and why is it popular?"
    result = ask(q)
    print(f"\nQuestion: {result['question']}\n")
    print("Tools used:")
    for c in result["tool_calls"]:
        print(f"  -> {c['tool']}({c['args']})")
    print(f"\nAnswer:\n{result['answer']}")
    print(f"\n({result['latency_s']}s)")
