"""Zero-cost guard: cloud AI calls stop at daily ceilings below the free plans' quotas."""
import pytest
from conftest import auth_headers  # noqa: F401  (fixture)

from app.ai import providers
from app.core.config import get_settings


@pytest.fixture
def cloud(monkeypatch):
    """Use the OpenAI-compatible providers with a stubbed network; returns the call log."""
    s = get_settings()
    monkeypatch.setattr(s, "openai_api_key", "k")
    calls = []

    def fake_post(url, payload, headers=None, timeout=60.0):
        calls.append(url)
        if url.endswith("/embeddings"):
            return {"data": [{"embedding": [1.0, 0.0]} for _ in payload["input"]]}
        return {"choices": [{"message": {"content": "Use neem oil [1]."}}], "usage": {"total_tokens": 1000}}

    monkeypatch.setattr(providers, "_post", fake_post)
    return s, calls


def test_chat_stops_at_the_daily_request_ceiling_without_calling_the_provider(cloud, monkeypatch):
    s, calls = cloud
    monkeypatch.setattr(s, "ai_daily_chat_requests", 3)
    llm = providers.OpenAILLM(s)
    for _ in range(3):
        llm.chat("sys", "q")
    with pytest.raises(providers.DailyBudgetExhausted):
        llm.chat("sys", "q")
    assert len(calls) == 3  # the refused call never reached the network
    assert providers.usage_today("chat") == {"requests": 3, "tokens": 3000}


def test_chat_stops_once_the_daily_token_ceiling_is_used(cloud, monkeypatch):
    s, calls = cloud
    monkeypatch.setattr(s, "ai_daily_chat_tokens", 2500)
    llm = providers.OpenAILLM(s)
    llm.chat("sys", "q")
    llm.chat("sys", "q")
    llm.chat("sys", "q")  # 2,000 used < 2,500: allowed, lands at 3,000
    with pytest.raises(providers.DailyBudgetExhausted):
        llm.chat("sys", "q")
    assert len(calls) == 3


def test_embeddings_have_their_own_ceiling(cloud, monkeypatch):
    s, calls = cloud
    monkeypatch.setattr(s, "ai_daily_embed_requests", 2)
    emb = providers.OpenAIEmbedder(s)
    emb.embed(["a"])
    emb.embed(["b"])
    with pytest.raises(providers.DailyBudgetExhausted):
        emb.embed(["c"])
    assert len(calls) == 2


def test_zero_switches_a_provider_off(cloud, monkeypatch):
    s, calls = cloud
    monkeypatch.setattr(s, "ai_daily_chat_requests", 0)
    with pytest.raises(providers.DailyBudgetExhausted):
        providers.OpenAILLM(s).chat("sys", "q")
    assert calls == []


def test_tokens_are_estimated_when_the_provider_reports_no_usage(cloud, monkeypatch):
    s, _ = cloud
    monkeypatch.setattr(providers, "_post", lambda *a, **k: {"choices": [{"message": {"content": "x" * 300}}]})
    providers.OpenAILLM(s).chat("s" * 300, "u" * 300)
    assert providers.usage_today("chat")["tokens"] == 300


def test_defaults_stay_below_the_free_plans(cloud):
    s, _ = cloud
    assert 0 < s.ai_daily_chat_requests < 1000 and 0 < s.ai_daily_chat_tokens < 200_000
    assert 0 < s.ai_daily_embed_requests < 1000


def test_assistant_explains_the_daily_free_limit(client, auth_headers, monkeypatch):  # noqa: F811
    from app.ai import rag

    def exhausted(*a, **k):
        raise providers.DailyBudgetExhausted("used up")
    monkeypatch.setattr(rag, "answer", exhausted)
    r = client.post("/api/v1/assistant/ask", headers=auth_headers, json={"question": "How to grow rice?"})
    assert r.status_code == 503 and "free today" in r.json()["detail"]
