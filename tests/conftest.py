import os
from types import SimpleNamespace

# Must happen before `main` (or anything importing it) gets imported anywhere in the
# test session — main.py raises RuntimeError at import time if these are unset.
os.environ.setdefault("GROQ_API_KEY", "test-groq-key")
os.environ.setdefault("TAVILY_API_KEY", "test-tavily-key")
os.environ.setdefault("API_KEY", "test-shared-key")

import pytest  # noqa: E402


def make_groq_response(content, total_tokens=100, prompt_tokens=60, completion_tokens=40):
    """Build a stand-in for a Groq chat-completion response — just the attributes
    main.py actually reads (usage.*, choices[0].message.content)."""
    return SimpleNamespace(
        usage=SimpleNamespace(
            total_tokens=total_tokens,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        ),
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
    )


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    """Point db.py at a throwaway SQLite file so tests never touch the real runs.db."""
    import db

    db_path = str(tmp_path / "test_runs.db")
    monkeypatch.setattr(db, "DB_PATH", db_path)
    db.init_db()
    return db
