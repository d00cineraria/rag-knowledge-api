// Package store はワーカーが使うSQLiteアクセスをまとめる。
//
// SQLiteドライバは github.com/ncruces/go-sqlite3（cgo不要のWASMビルド）を使い、
// sqlite-vec拡張は github.com/asg017/sqlite-vec-go-bindings/ncruces が提供する
// vec0対応済みのSQLiteバイナリに差し替える形で組み込む。
package store

import (
	"context"
	"database/sql"
	"errors"
	"fmt"

	vec "github.com/asg017/sqlite-vec-go-bindings/ncruces"
	_ "github.com/ncruces/go-sqlite3/driver" // database/sql に "sqlite3" ドライバを登録する
)

// ErrNoDocument は status='embedding' のdocumentが存在しないことを表す。
var ErrNoDocument = errors.New("store: no document ready for embedding")

// Document はワーカーがclaimしたdocumentレコード。
type Document struct {
	ID           string
	CollectionID string
}

// Chunk はembedding未登録のチャンク。
type Chunk struct {
	ID      string
	Content string
}

// Store はワーカー用のSQLiteアクセスをまとめた薄いラッパー。
type Store struct {
	db *sql.DB
}

// Open はSQLiteファイルを開く。
// busy_timeoutと排他ロック（_txlock=immediate）をDSNで指定し、
// 他プロセス（FastAPI側のPythonプロセス）との書き込み競合に対して安全にリトライさせる。
func Open(path string) (*Store, error) {
	dsn := fmt.Sprintf("file:%s?_pragma=busy_timeout(5000)&_txlock=immediate", path)
	db, err := sql.Open("sqlite3", dsn)
	if err != nil {
		return nil, fmt.Errorf("store: open %s: %w", path, err)
	}
	// SQLiteは単一ライターのため、ワーカー内の接続を1本に絞ってロック競合を減らす。
	db.SetMaxOpenConns(1)

	return &Store{db: db}, nil
}

// Close はDB接続を閉じる。
func (s *Store) Close() error {
	return s.db.Close()
}

// ClaimNextDocument はstatus='embedding'の最古のdocumentを1件、
// status='embedding_processing'へ原子的に更新して返す。
// 対象がなければErrNoDocumentを返す。
func (s *Store) ClaimNextDocument(ctx context.Context) (Document, error) {
	tx, err := s.db.BeginTx(ctx, nil)
	if err != nil {
		return Document{}, fmt.Errorf("claim next document: begin tx: %w", err)
	}
	defer tx.Rollback() //nolint:errcheck // Commit後のRollbackは無害（database/sqlの仕様）

	var doc Document
	err = tx.QueryRowContext(ctx, `
		SELECT id, collection_id FROM documents
		WHERE status = 'embedding'
		ORDER BY created_at ASC
		LIMIT 1
	`).Scan(&doc.ID, &doc.CollectionID)
	switch {
	case errors.Is(err, sql.ErrNoRows):
		return Document{}, ErrNoDocument
	case err != nil:
		return Document{}, fmt.Errorf("claim next document: select candidate: %w", err)
	}

	result, err := tx.ExecContext(ctx, `
		UPDATE documents SET status = 'embedding_processing', updated_at = datetime('now')
		WHERE id = ? AND status = 'embedding'
	`, doc.ID)
	if err != nil {
		return Document{}, fmt.Errorf("claim next document: update status: %w", err)
	}
	affected, err := result.RowsAffected()
	if err != nil {
		return Document{}, fmt.Errorf("claim next document: rows affected: %w", err)
	}
	if affected == 0 {
		// 同じ瞬間に他のワーカーが先に claim した場合（トランザクション分離により通常は起きない）。
		return Document{}, ErrNoDocument
	}

	if err := tx.Commit(); err != nil {
		return Document{}, fmt.Errorf("claim next document: commit: %w", err)
	}
	return doc, nil
}

// PendingChunks はdocumentIDに属し、まだchunk_vectorsに登録されていないチャンクを
// chunk_index順に返す。
func (s *Store) PendingChunks(ctx context.Context, documentID string) ([]Chunk, error) {
	rows, err := s.db.QueryContext(ctx, `
		SELECT c.id, c.content
		FROM chunks c
		LEFT JOIN chunk_vectors v ON v.chunk_id = c.id
		WHERE c.document_id = ? AND v.chunk_id IS NULL
		ORDER BY c.chunk_index ASC
	`, documentID)
	if err != nil {
		return nil, fmt.Errorf("pending chunks: query: %w", err)
	}
	defer rows.Close()

	var chunks []Chunk
	for rows.Next() {
		var c Chunk
		if err := rows.Scan(&c.ID, &c.Content); err != nil {
			return nil, fmt.Errorf("pending chunks: scan: %w", err)
		}
		chunks = append(chunks, c)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("pending chunks: iterate: %w", err)
	}
	return chunks, nil
}

// InsertVectors はchunkIDs[i]に対応するvectors[i]をchunk_vectorsへ書き込む。
// ベクトルはsqlite-vecバインディングのSerializeFloat32でバイト列化する
// （Python側sqlite_vec.serialize_float32と互換のフォーマット）。
func (s *Store) InsertVectors(ctx context.Context, chunkIDs []string, vectors [][]float32) error {
	if len(chunkIDs) != len(vectors) {
		return fmt.Errorf("insert vectors: length mismatch: %d ids, %d vectors", len(chunkIDs), len(vectors))
	}

	tx, err := s.db.BeginTx(ctx, nil)
	if err != nil {
		return fmt.Errorf("insert vectors: begin tx: %w", err)
	}
	defer tx.Rollback() //nolint:errcheck

	stmt, err := tx.PrepareContext(ctx, `INSERT INTO chunk_vectors (chunk_id, embedding) VALUES (?, ?)`)
	if err != nil {
		return fmt.Errorf("insert vectors: prepare: %w", err)
	}
	defer stmt.Close()

	for i, id := range chunkIDs {
		blob, err := vec.SerializeFloat32(vectors[i])
		if err != nil {
			return fmt.Errorf("insert vectors: serialize chunk %s: %w", id, err)
		}
		if _, err := stmt.ExecContext(ctx, id, blob); err != nil {
			return fmt.Errorf("insert vectors: exec chunk %s: %w", id, err)
		}
	}

	if err := tx.Commit(); err != nil {
		return fmt.Errorf("insert vectors: commit: %w", err)
	}
	return nil
}

// MarkReady はdocumentをstatus='ready'に更新する。
func (s *Store) MarkReady(ctx context.Context, documentID string) error {
	_, err := s.db.ExecContext(ctx, `
		UPDATE documents SET status = 'ready', error = NULL, updated_at = datetime('now')
		WHERE id = ?
	`, documentID)
	if err != nil {
		return fmt.Errorf("mark ready: %w", err)
	}
	return nil
}

// MarkError はdocumentをstatus='error'に更新し、エラーメッセージを記録する。
// メッセージは2000文字（Unicode文字数）までに切り詰める。
func (s *Store) MarkError(ctx context.Context, documentID string, cause error) error {
	_, err := s.db.ExecContext(ctx, `
		UPDATE documents SET status = 'error', error = ?, updated_at = datetime('now')
		WHERE id = ?
	`, truncate(cause.Error(), 2000), documentID)
	if err != nil {
		return fmt.Errorf("mark error: %w", err)
	}
	return nil
}

func truncate(s string, max int) string {
	runes := []rune(s)
	if len(runes) <= max {
		return s
	}
	return string(runes[:max])
}
