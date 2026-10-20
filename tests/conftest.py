"""Shared test setup. Runs automatically before any test."""
import os

from dotenv import load_dotenv

load_dotenv()
# None of the tests call Claude for real, but the agent needs SOME key to start up.
os.environ.setdefault("ANTHROPIC_API_KEY", "test-key")
