# Agentic Research Assistant

An agentic LLM app that answers questions about research papers using Claude, LangGraph, and a RAG pipeline over Chroma.

## Setup
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env        # then paste your API key in .env
python hello_claude.py
```
