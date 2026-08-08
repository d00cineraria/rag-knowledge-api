package config

import (
	"testing"
	"time"
)

func TestLoadDefaults(t *testing.T) {
	t.Setenv("GEMINI_API_KEY", "test-key")
	t.Setenv("LLM_PROVIDER", "")
	t.Setenv("SQLITE_PATH", "")
	t.Setenv("GEMINI_EMBED_MODEL", "")
	t.Setenv("OLLAMA_BASE_URL", "")
	t.Setenv("OLLAMA_EMBED_MODEL", "")
	t.Setenv("EMBED_DIM", "")
	t.Setenv("POLL_INTERVAL_SECONDS", "")

	cfg, err := Load()
	if err != nil {
		t.Fatalf("Load() returned error: %v", err)
	}

	if cfg.SQLitePath != "./data/rag.db" {
		t.Errorf("SQLitePath = %q, want default", cfg.SQLitePath)
	}
	if cfg.LLMProvider != "ollama" {
		t.Errorf("LLMProvider = %q, want ollama", cfg.LLMProvider)
	}
	if cfg.GeminiEmbedModel != "gemini-embedding-001" {
		t.Errorf("GeminiEmbedModel = %q, want default", cfg.GeminiEmbedModel)
	}
	if cfg.OllamaBaseURL != "http://localhost:11434" {
		t.Errorf("OllamaBaseURL = %q, want default", cfg.OllamaBaseURL)
	}
	if cfg.OllamaEmbedModel != "nomic-embed-text" {
		t.Errorf("OllamaEmbedModel = %q, want default", cfg.OllamaEmbedModel)
	}
	if cfg.EmbedDim != 768 {
		t.Errorf("EmbedDim = %d, want 768", cfg.EmbedDim)
	}
	if cfg.PollInterval != 2*time.Second {
		t.Errorf("PollInterval = %v, want 2s", cfg.PollInterval)
	}
}

func TestLoadOverrides(t *testing.T) {
	t.Setenv("GEMINI_API_KEY", "test-key")
	t.Setenv("SQLITE_PATH", "/tmp/custom.db")
	t.Setenv("GEMINI_EMBED_MODEL", "custom-model")
	t.Setenv("EMBED_DIM", "1536")
	t.Setenv("POLL_INTERVAL_SECONDS", "5")

	cfg, err := Load()
	if err != nil {
		t.Fatalf("Load() returned error: %v", err)
	}

	if cfg.SQLitePath != "/tmp/custom.db" {
		t.Errorf("SQLitePath = %q, want override", cfg.SQLitePath)
	}
	if cfg.EmbedDim != 1536 {
		t.Errorf("EmbedDim = %d, want 1536", cfg.EmbedDim)
	}
	if cfg.PollInterval != 5*time.Second {
		t.Errorf("PollInterval = %v, want 5s", cfg.PollInterval)
	}
}

func TestLoadMissingAPIKeyWithGeminiProvider(t *testing.T) {
	t.Setenv("LLM_PROVIDER", "gemini")
	t.Setenv("GEMINI_API_KEY", "")

	if _, err := Load(); err == nil {
		t.Fatal("Load() expected error for missing GEMINI_API_KEY when LLM_PROVIDER=gemini, got nil")
	}
}

func TestLoadMissingAPIKeyWithOllamaProviderSucceeds(t *testing.T) {
	t.Setenv("LLM_PROVIDER", "ollama")
	t.Setenv("GEMINI_API_KEY", "")

	cfg, err := Load()
	if err != nil {
		t.Fatalf("Load() returned error: %v", err)
	}
	if cfg.LLMProvider != "ollama" {
		t.Errorf("LLMProvider = %q, want ollama", cfg.LLMProvider)
	}
}

func TestLoadDefaultProviderSucceedsWithoutAPIKey(t *testing.T) {
	t.Setenv("LLM_PROVIDER", "")
	t.Setenv("GEMINI_API_KEY", "")

	if _, err := Load(); err != nil {
		t.Fatalf("Load() returned error for default provider without GEMINI_API_KEY: %v", err)
	}
}

func TestLoadInvalidInt(t *testing.T) {
	t.Setenv("GEMINI_API_KEY", "test-key")
	t.Setenv("EMBED_DIM", "not-a-number")

	if _, err := Load(); err == nil {
		t.Fatal("Load() expected error for invalid EMBED_DIM, got nil")
	}
}
