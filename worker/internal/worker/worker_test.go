package worker

import (
	"context"
	"database/sql"
	"errors"
	"io"
	"log"
	"path/filepath"
	"testing"
	"time"

	"github.com/d00cineraria/rag-knowledge-api/worker/internal/store"
)

const testSchema = `
	CREATE TABLE documents (
		id TEXT PRIMARY KEY,
		collection_id TEXT NOT NULL,
		filename TEXT NOT NULL,
		status TEXT NOT NULL DEFAULT 'pending',
		error TEXT,
		created_at TEXT NOT NULL DEFAULT (datetime('now')),
		updated_at TEXT NOT NULL DEFAULT (datetime('now'))
	);
	CREATE TABLE chunks (
		id TEXT PRIMARY KEY,
		document_id TEXT NOT NULL,
		collection_id TEXT NOT NULL,
		chunk_index INTEGER NOT NULL,
		content TEXT NOT NULL
	);
	CREATE VIRTUAL TABLE chunk_vectors USING vec0(chunk_id TEXT PRIMARY KEY, embedding float[3]);
`

// fakeEmbedder はGemini呼び出しをモックする。
type fakeEmbedder struct {
	dim  int
	fail bool
}

func (f *fakeEmbedder) Embed(_ context.Context, texts []string) ([][]float32, error) {
	if f.fail {
		return nil, errors.New("embedding failed (mock)")
	}
	out := make([][]float32, len(texts))
	for i := range texts {
		v := make([]float32, f.dim)
		v[0] = 1
		out[i] = v
	}
	return out, nil
}

// newTestStore は一時SQLiteファイルにスキーマと初期データを流し込み、store.Storeでラップして返す。
// store.Storeはdb接続をカプセル化しているため、seedSQLの実行だけは別接続（database/sql直接）で行う。
func newTestStore(t *testing.T, seedSQL string) (*store.Store, string) {
	t.Helper()

	dbPath := filepath.Join(t.TempDir(), "test.db")

	setup, err := sql.Open("sqlite3", "file:"+dbPath)
	if err != nil {
		t.Fatalf("open setup connection: %v", err)
	}
	if _, err := setup.Exec(testSchema + seedSQL); err != nil {
		t.Fatalf("seed schema/data: %v", err)
	}
	if err := setup.Close(); err != nil {
		t.Fatalf("close setup connection: %v", err)
	}

	st, err := store.Open(dbPath)
	if err != nil {
		t.Fatalf("store.Open() error = %v", err)
	}
	t.Cleanup(func() { _ = st.Close() })
	return st, dbPath
}

func queryDocumentStatus(t *testing.T, dbPath, documentID string) (status string, errMsg sql.NullString) {
	t.Helper()

	conn, err := sql.Open("sqlite3", "file:"+dbPath)
	if err != nil {
		t.Fatalf("open verify connection: %v", err)
	}
	defer conn.Close()

	if err := conn.QueryRow(`SELECT status, error FROM documents WHERE id = ?`, documentID).
		Scan(&status, &errMsg); err != nil {
		t.Fatalf("query status: %v", err)
	}
	return status, errMsg
}

func TestWorkerClaimToReadyIntegration(t *testing.T) {
	st, dbPath := newTestStore(t, `
		INSERT INTO documents (id, collection_id, filename, status) VALUES ('doc-1', 'col-1', 'doc.md', 'embedding');
		INSERT INTO chunks (id, document_id, collection_id, chunk_index, content) VALUES
			('chunk-1', 'doc-1', 'col-1', 0, 'first chunk'),
			('chunk-2', 'doc-1', 'col-1', 1, 'second chunk');
	`)

	logger := log.New(io.Discard, "", 0)
	w := New(st, &fakeEmbedder{dim: 3}, time.Second, logger)

	w.tick(t.Context())

	status, errMsg := queryDocumentStatus(t, dbPath, "doc-1")
	if status != "ready" {
		t.Fatalf("status = %q, want ready (error=%v)", status, errMsg)
	}

	verify, err := sql.Open("sqlite3", "file:"+dbPath)
	if err != nil {
		t.Fatalf("open verify connection: %v", err)
	}
	defer verify.Close()

	var vectorCount int
	if err := verify.QueryRow(`SELECT COUNT(*) FROM chunk_vectors`).Scan(&vectorCount); err != nil {
		t.Fatalf("count chunk_vectors: %v", err)
	}
	if vectorCount != 2 {
		t.Fatalf("chunk_vectors count = %d, want 2", vectorCount)
	}
}

func TestWorkerMarksErrorOnEmbedFailure(t *testing.T) {
	st, dbPath := newTestStore(t, `
		INSERT INTO documents (id, collection_id, filename, status) VALUES ('doc-1', 'col-1', 'doc.md', 'embedding');
		INSERT INTO chunks (id, document_id, collection_id, chunk_index, content) VALUES
			('chunk-1', 'doc-1', 'col-1', 0, 'first chunk');
	`)

	logger := log.New(io.Discard, "", 0)
	w := New(st, &fakeEmbedder{dim: 3, fail: true}, time.Second, logger)

	w.tick(t.Context())

	status, errMsg := queryDocumentStatus(t, dbPath, "doc-1")
	if status != "error" {
		t.Fatalf("status = %q, want error", status)
	}
	if !errMsg.Valid || errMsg.String == "" {
		t.Fatalf("error message = %v, want non-empty", errMsg)
	}
}

func TestWorkerTickNoopWhenNothingToClaim(t *testing.T) {
	st, dbPath := newTestStore(t, `
		INSERT INTO documents (id, collection_id, filename, status) VALUES ('doc-1', 'col-1', 'doc.md', 'ready');
	`)

	logger := log.New(io.Discard, "", 0)
	w := New(st, &fakeEmbedder{dim: 3}, time.Second, logger)

	w.tick(t.Context())

	status, _ := queryDocumentStatus(t, dbPath, "doc-1")
	if status != "ready" {
		t.Fatalf("status = %q, want unchanged ready", status)
	}
}
