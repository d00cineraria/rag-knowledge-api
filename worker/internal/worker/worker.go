// Package worker はSQLiteをポーリングし、embedding待ちのdocumentを処理するループを提供する。
package worker

import (
	"context"
	"errors"
	"fmt"
	"log"
	"time"

	"github.com/d00cineraria/rag-knowledge-api/worker/internal/store"
)

// Embedder はテキストをベクトル化する。internal/embedding.Client が実装する。
// HTTP呼び出しがテスト時にモックしやすいよう、境界をインターフェースにしている。
type Embedder interface {
	Embed(ctx context.Context, texts []string) ([][]float32, error)
}

// Worker はclaim→embed→readyのポーリングループを実行する。
type Worker struct {
	store        *store.Store
	embedder     Embedder
	pollInterval time.Duration
	logger       *log.Logger
}

// New はWorkerを構築する。
func New(s *store.Store, e Embedder, pollInterval time.Duration, logger *log.Logger) *Worker {
	return &Worker{store: s, embedder: e, pollInterval: pollInterval, logger: logger}
}

// Run はctxがキャンセルされるまでポーリングを繰り返す。
// SIGINT/SIGTERM時はctx側（main.goのsignal.NotifyContext）が閉じることで
// graceful shutdownする。
func (w *Worker) Run(ctx context.Context) {
	ticker := time.NewTicker(w.pollInterval)
	defer ticker.Stop()

	for {
		w.tick(ctx)

		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
		}
	}
}

// tick は1回分のポーリング処理: 1件claimし、処理結果に応じてready/errorへ更新する。
// 対象がない場合は何もしない。
func (w *Worker) tick(ctx context.Context) {
	doc, err := w.store.ClaimNextDocument(ctx)
	if errors.Is(err, store.ErrNoDocument) {
		return
	}
	if err != nil {
		w.logger.Printf("worker: claim failed: %v", err)
		return
	}

	if err := w.processDocument(ctx, doc); err != nil {
		w.logger.Printf("worker: document %s failed: %v", doc.ID, err)
		if markErr := w.store.MarkError(ctx, doc.ID, err); markErr != nil {
			w.logger.Printf("worker: mark error failed for document %s: %v", doc.ID, markErr)
		}
		return
	}
	w.logger.Printf("worker: document %s ready", doc.ID)
}

// processDocument はclaim済みdocumentのチャンクをembeddingし、chunk_vectorsへ書き込む。
func (w *Worker) processDocument(ctx context.Context, doc store.Document) error {
	chunks, err := w.store.PendingChunks(ctx, doc.ID)
	if err != nil {
		return fmt.Errorf("pending chunks: %w", err)
	}
	if len(chunks) == 0 {
		return w.store.MarkReady(ctx, doc.ID)
	}

	texts := make([]string, len(chunks))
	ids := make([]string, len(chunks))
	for i, c := range chunks {
		texts[i] = c.Content
		ids[i] = c.ID
	}

	vectors, err := w.embedder.Embed(ctx, texts)
	if err != nil {
		return fmt.Errorf("embed: %w", err)
	}
	if len(vectors) != len(ids) {
		return fmt.Errorf("embed: got %d vectors for %d chunks", len(vectors), len(ids))
	}

	if err := w.store.InsertVectors(ctx, ids, vectors); err != nil {
		return fmt.Errorf("insert vectors: %w", err)
	}

	return w.store.MarkReady(ctx, doc.ID)
}
