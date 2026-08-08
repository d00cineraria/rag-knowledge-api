package store

import (
	"database/sql"
	"errors"
	"path/filepath"
	"testing"
)

// openTestStore はテスト専用の一時SQLiteファイルにPython側db.pyと同等のスキーマを作り、
// Storeでラップして返す。
func openTestStore(t *testing.T) *Store {
	t.Helper()

	dbPath := filepath.Join(t.TempDir(), "test.db")
	st, err := Open(dbPath)
	if err != nil {
		t.Fatalf("Open() error = %v", err)
	}
	t.Cleanup(func() { _ = st.Close() })

	schema := `
		CREATE TABLE collections (
			id TEXT PRIMARY KEY,
			name TEXT NOT NULL UNIQUE
		);
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
			content TEXT NOT NULL,
			heading_path TEXT NOT NULL DEFAULT '[]',
			token_count INTEGER
		);
		CREATE VIRTUAL TABLE chunk_vectors USING vec0(chunk_id TEXT PRIMARY KEY, embedding float[3]);
	`
	if _, err := st.db.Exec(schema); err != nil {
		t.Fatalf("create schema: %v", err)
	}
	return st
}

func insertDocument(t *testing.T, st *Store, id, status string) {
	t.Helper()
	if _, err := st.db.Exec(
		`INSERT INTO collections (id, name) VALUES (?, ?)`, "col-"+id, "col-"+id,
	); err != nil {
		t.Fatalf("insert collection: %v", err)
	}
	if _, err := st.db.Exec(
		`INSERT INTO documents (id, collection_id, filename, status) VALUES (?, ?, 'doc.md', ?)`,
		id, "col-"+id, status,
	); err != nil {
		t.Fatalf("insert document: %v", err)
	}
}

func insertChunk(t *testing.T, st *Store, id, documentID string, index int, content string) {
	t.Helper()
	_, err := st.db.Exec(
		`INSERT INTO chunks (id, document_id, collection_id, chunk_index, content) VALUES (?, ?, ?, ?, ?)`,
		id, documentID, "col-"+documentID, index, content,
	)
	if err != nil {
		t.Fatalf("insert chunk: %v", err)
	}
}

func TestClaimNextDocumentReturnsOldestEmbeddingDocument(t *testing.T) {
	st := openTestStore(t)
	ctx := t.Context()

	insertDocument(t, st, "doc-pending", "pending")
	insertDocument(t, st, "doc-embedding", "embedding")

	doc, err := st.ClaimNextDocument(ctx)
	if err != nil {
		t.Fatalf("ClaimNextDocument() error = %v", err)
	}
	if doc.ID != "doc-embedding" {
		t.Fatalf("claimed document = %q, want doc-embedding", doc.ID)
	}

	var status string
	if err := st.db.QueryRow(`SELECT status FROM documents WHERE id = ?`, doc.ID).Scan(&status); err != nil {
		t.Fatalf("query status: %v", err)
	}
	if status != "embedding_processing" {
		t.Fatalf("status after claim = %q, want embedding_processing", status)
	}
}

func TestClaimNextDocumentNoCandidate(t *testing.T) {
	st := openTestStore(t)
	ctx := t.Context()

	insertDocument(t, st, "doc-ready", "ready")

	_, err := st.ClaimNextDocument(ctx)
	if !errors.Is(err, ErrNoDocument) {
		t.Fatalf("ClaimNextDocument() error = %v, want ErrNoDocument", err)
	}
}

func TestPendingChunksExcludesAlreadyEmbedded(t *testing.T) {
	st := openTestStore(t)
	ctx := t.Context()

	insertDocument(t, st, "doc-1", "embedding_processing")
	insertChunk(t, st, "chunk-1", "doc-1", 0, "first")
	insertChunk(t, st, "chunk-2", "doc-1", 1, "second")

	if err := st.InsertVectors(ctx, []string{"chunk-1"}, [][]float32{{1, 0, 0}}); err != nil {
		t.Fatalf("InsertVectors() error = %v", err)
	}

	chunks, err := st.PendingChunks(ctx, "doc-1")
	if err != nil {
		t.Fatalf("PendingChunks() error = %v", err)
	}
	if len(chunks) != 1 || chunks[0].ID != "chunk-2" {
		t.Fatalf("PendingChunks() = %+v, want only chunk-2", chunks)
	}
}

func TestMarkReadyAndMarkError(t *testing.T) {
	st := openTestStore(t)
	ctx := t.Context()

	insertDocument(t, st, "doc-1", "embedding_processing")

	if err := st.MarkReady(ctx, "doc-1"); err != nil {
		t.Fatalf("MarkReady() error = %v", err)
	}
	var status string
	var errMsg sql.NullString
	if err := st.db.QueryRow(`SELECT status, error FROM documents WHERE id = ?`, "doc-1").
		Scan(&status, &errMsg); err != nil {
		t.Fatalf("query: %v", err)
	}
	if status != "ready" || errMsg.Valid {
		t.Fatalf("after MarkReady: status=%q error=%v, want ready/NULL", status, errMsg)
	}

	if err := st.MarkError(ctx, "doc-1", errors.New("boom")); err != nil {
		t.Fatalf("MarkError() error = %v", err)
	}
	if err := st.db.QueryRow(`SELECT status, error FROM documents WHERE id = ?`, "doc-1").
		Scan(&status, &errMsg); err != nil {
		t.Fatalf("query: %v", err)
	}
	if status != "error" || errMsg.String != "boom" {
		t.Fatalf("after MarkError: status=%q error=%q, want error/boom", status, errMsg.String)
	}
}

func TestMarkErrorTruncatesLongMessage(t *testing.T) {
	st := openTestStore(t)
	ctx := t.Context()
	insertDocument(t, st, "doc-1", "embedding_processing")

	longMsg := make([]rune, 3000)
	for i := range longMsg {
		longMsg[i] = 'あ'
	}

	if err := st.MarkError(ctx, "doc-1", errors.New(string(longMsg))); err != nil {
		t.Fatalf("MarkError() error = %v", err)
	}

	var errMsg string
	if err := st.db.QueryRow(`SELECT error FROM documents WHERE id = ?`, "doc-1").Scan(&errMsg); err != nil {
		t.Fatalf("query: %v", err)
	}
	if got := len([]rune(errMsg)); got != 2000 {
		t.Fatalf("truncated error length = %d runes, want 2000", got)
	}
}

func TestInsertVectorsLengthMismatch(t *testing.T) {
	st := openTestStore(t)

	err := st.InsertVectors(t.Context(), []string{"a", "b"}, [][]float32{{1, 0, 0}})
	if err == nil {
		t.Fatal("InsertVectors() expected error for length mismatch, got nil")
	}
}
