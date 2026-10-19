"""
Quick test of the RAG index: ask a question, see which chunks come back.

Run from the project root:
    python try_search.py "your question here"
"""
import sys

import chromadb

question = sys.argv[1] if len(sys.argv) > 1 else "How do quantum error correcting codes work?"

client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_collection("papers")

results = collection.query(query_texts=[question], n_results=3)

print(f"Question: {question}\n")
for doc, meta, dist in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
    print(f"--- {meta['title']} (arXiv {meta['paper_id']}, page {meta['page']})")
    print(f"    distance: {dist:.3f}  (lower = closer in meaning)")
    print(f"    {doc[:300]}...\n")
