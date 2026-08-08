// Command worker はPythonのAPIサーバーと同じSQLiteファイルをポーリングし、
// status='embedding' のdocumentに対してGemini embeddingを実行するバックグラウンドワーカー。
//
// 詳しい役割・起動方法・Go学習ノートはREADME.mdを参照。
package main

import (
	"context"
	"log"
	"os"
	"os/signal"
	"syscall"

	"github.com/d00cineraria/rag-knowledge-api/worker/internal/config"
	"github.com/d00cineraria/rag-knowledge-api/worker/internal/embedding"
	"github.com/d00cineraria/rag-knowledge-api/worker/internal/store"
	"github.com/d00cineraria/rag-knowledge-api/worker/internal/worker"
)

func main() {
	logger := log.New(os.Stdout, "", log.LstdFlags)

	cfg, err := config.Load()
	if err != nil {
		logger.Fatalf("config: %v", err)
	}

	st, err := store.Open(cfg.SQLitePath)
	if err != nil {
		logger.Fatalf("store: %v", err)
	}
	defer st.Close()

	embedder := embedding.NewClient(cfg.GeminiAPIKey, cfg.GeminiEmbedModel, cfg.EmbedDim)
	w := worker.New(st, embedder, cfg.PollInterval, logger)

	// SIGINT/SIGTERMを受けたらctxを閉じ、ポーリングループを止める（graceful shutdown）。
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	logger.Printf(
		"worker: starting (db=%s, model=%s, dim=%d, poll=%s)",
		cfg.SQLitePath, cfg.GeminiEmbedModel, cfg.EmbedDim, cfg.PollInterval,
	)
	w.Run(ctx)
	logger.Println("worker: stopped")
}
