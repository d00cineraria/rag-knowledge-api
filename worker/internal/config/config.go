// Package config は環境変数からワーカーの実行設定を読み込む。
package config

import (
	"fmt"
	"os"
	"strconv"
	"time"
)

// Config はワーカー起動に必要な設定値。
type Config struct {
	SQLitePath       string
	GeminiAPIKey     string
	GeminiEmbedModel string
	EmbedDim         int
	PollInterval     time.Duration
}

// Load は環境変数からConfigを組み立てる。GEMINI_API_KEYが空だとエラーを返す。
func Load() (Config, error) {
	embedDim, err := envInt("EMBED_DIM", 768)
	if err != nil {
		return Config{}, err
	}

	pollSeconds, err := envInt("POLL_INTERVAL_SECONDS", 2)
	if err != nil {
		return Config{}, err
	}

	cfg := Config{
		SQLitePath:       envString("SQLITE_PATH", "./data/rag.db"),
		GeminiAPIKey:     os.Getenv("GEMINI_API_KEY"),
		GeminiEmbedModel: envString("GEMINI_EMBED_MODEL", "gemini-embedding-001"),
		EmbedDim:         embedDim,
		PollInterval:     time.Duration(pollSeconds) * time.Second,
	}

	if cfg.GeminiAPIKey == "" {
		return Config{}, fmt.Errorf("config: GEMINI_API_KEY is required")
	}

	return cfg, nil
}

func envString(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}

func envInt(key string, fallback int) (int, error) {
	raw := os.Getenv(key)
	if raw == "" {
		return fallback, nil
	}
	v, err := strconv.Atoi(raw)
	if err != nil {
		return 0, fmt.Errorf("config: invalid %s=%q: %w", key, raw, err)
	}
	return v, nil
}
