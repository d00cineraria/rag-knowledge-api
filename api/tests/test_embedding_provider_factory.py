"""get_embedding_provider()がsettings.llm_providerに応じて正しい実装を返すことを検証する。"""

from app.config import settings
from app.services import embedding


def test_returns_ollama_provider_by_default(monkeypatch):
    monkeypatch.setattr(settings, "llm_provider", "ollama")
    provider = embedding.get_embedding_provider()
    assert isinstance(provider, embedding.OllamaEmbeddingProvider)


def test_returns_gemini_provider_when_configured(monkeypatch):
    monkeypatch.setattr(settings, "llm_provider", "gemini")
    monkeypatch.setattr(settings, "gemini_api_key", "test-key")
    provider = embedding.get_embedding_provider()
    assert isinstance(provider, embedding.GeminiEmbeddingProvider)
