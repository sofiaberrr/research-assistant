"""Sanity check: confirms your API key works and Claude responds."""
import os

from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()  # reads ANTHROPIC_API_KEY from .env

client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

response = client.messages.create(
    model="claude-sonnet-5-5",
    max_tokens=200,
    messages=[{"role": "user", "content": "In one sentence, what is retrieval-augmented generation?"}],
)

print(response.content[0].text)
