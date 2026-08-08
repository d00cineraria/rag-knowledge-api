# worker — 取り込みワーカー (Go)

Python API (`api/`) と同じSQLiteファイルをポーリングし、`status='embedding'` の
documentに対してembeddingを実行し、`chunk_vectors` へ書き込むバックグラウンド
プロセス。チャンキング自体はPython側（`api/app/services/ingest`）に一本化されており、
このワーカーは「チャンク本文 → ベクトル」の変換だけを担う。embeddingプロバイダは
Ollama（既定・ローカル・APIキー不要）とGemini（クラウド）を`LLM_PROVIDER`で切り替えられる。

## 役割

1. `INGEST_MODE=worker` のときPythonの `process_document` は、パース→チャンキング→
   `chunks`/`chunks_fts` へのINSERTまでを行い、`documents.status='embedding'` で処理を止める
2. このワーカーが `status='embedding'` のdocumentを2秒間隔（既定）でポーリングし、
   1件を原子的にclaim（`status='embedding_processing'` へ更新）する
3. そのdocumentの未embeddingチャンクをまとめて設定されたプロバイダ（Ollama `/api/embed`
   またはGemini `batchEmbedContents`）に投げ、L2正規化してから `chunk_vectors` へ書き込む
4. 成功したら `status='ready'`、失敗したら `status='error'` + エラーメッセージ（最大2000字）

`INGEST_MODE=inline`（既定）のときはPython側がembeddingまで完結するため、このワーカーは
不要（クイックスタートを壊さないための設計）。

## 起動方法

```bash
# 依存取得（初回のみ）
go mod download

# テスト
go vet ./...
go test ./...

# 起動（api/ 側と同じSQLITE_PATHを指すこと。既定プロバイダはollama）
SQLITE_PATH=../api/data/rag.db \
go run .

# Geminiプロバイダを使う場合
SQLITE_PATH=../api/data/rag.db \
LLM_PROVIDER=gemini \
GEMINI_API_KEY=... \
go run .
```

### 環境変数

| 変数 | 既定値 | 説明 |
|---|---|---|
| `SQLITE_PATH` | `./data/rag.db` | Python APIと共有するSQLiteファイル |
| `LLM_PROVIDER` | `ollama` | embeddingプロバイダ（`ollama` \| `gemini`） |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | OllamaのベースURL（`LLM_PROVIDER=ollama`時） |
| `OLLAMA_EMBED_MODEL` | `nomic-embed-text` | Ollama embeddingモデル名 |
| `GEMINI_API_KEY` | (未設定可) | Gemini embedding APIキー。`LLM_PROVIDER=gemini`のときのみ必須 |
| `GEMINI_EMBED_MODEL` | `gemini-embedding-001` | Gemini embeddingモデル名 |
| `EMBED_DIM` | `768` | 出力次元（Gemini `outputDimensionality`。Ollama側はモデル固有の次元をそのまま使う） |
| `POLL_INTERVAL_SECONDS` | `2` | ポーリング間隔（秒） |

`Ctrl-C` (SIGINT) または SIGTERM で graceful shutdown する（処理中のドキュメントは
中断されず、次のポーリング前にループを抜ける）。

## 構成

```
worker/
├── main.go                    # エントリーポイント。設定読込→プロバイダ選択→DB接続→シグナル待受
└── internal/
    ├── config/                # 環境変数の読み込み
    ├── embedding/              # Gemini batchEmbedContents / Ollama /api/embed クライアント + L2正規化
    ├── store/                  # SQLite操作（claim / pending chunks / insert / mark ready|error）
    └── worker/                 # ポーリングループの本体（store と embedding を組み合わせる）
```

`internal/` はGoの言語仕様として、モジュール外（`github.com/...`）から一切importできない
ディレクトリ。ワーカーの実装詳細を外部に晒したくないので採用している。

## Go学習ノート

このリポジトリのオーナー（PHP/TypeScript/Python経験者）向けに、実装で実際に使った
Goの言語機能・標準ライブラリの使い方を機能ごとに解説する。

### 1. パッケージとディレクトリ構成

Goは1ディレクトリ = 1パッケージ。`import "github.com/.../worker/internal/store"` の
ように、ディレクトリパスがそのままimportパスになる。PHPのnamespaceやTSの
`import from "./store"` に近いが、ファイル単位ではなくディレクトリ単位。

### 2. 構造体 (struct) とメソッド

`type Store struct { db *sql.DB }` のように、関連するデータをまとめる型。
PHPのクラスに近いが、継承はなくコンポジション（構造体埋め込み）を使う。
`func (s *Store) Open(...)` のように、型に対して関数（メソッド）を後付けで定義する。

### 3. ポインタレシーバ vs 値レシーバ

`func (s *Store) Close() error` の `*Store` はポインタレシーバ。メソッド内で
フィールドを書き換えたい場合や、構造体のコピーコストを避けたい場合に使う
（`store.go` のメソッドは全てポインタレシーバ）。値レシーバは呼び出し時にコピーが
渡るため、読み取り専用の小さな値向き。

### 4. インターフェース (interface)

`internal/worker/worker.go` の `Embedder` インターフェースは
`Embed(ctx, texts) ([][]float32, error)` という1メソッドのシグネチャだけを定義する。
`internal/embedding.Client`（Gemini）と `internal/embedding.OllamaClient`（Ollama）は
どちらもこのメソッドを実装しているというだけで、明示的な `implements` 宣言なしに
`Embedder` として扱える（構造的型付け）。`main.go` は `LLM_PROVIDER` の値に応じて
どちらの実装を `worker.New` に渡すか切り替えているだけで、`worker.go`・`store.go` は
プロバイダの違いを一切知らない。テストでは `fakeEmbedder` を代わりに渡すことで、
HTTP通信なしにワーカーのロジックだけ検証できる。

### 5. エラーハンドリング: `error` 型と `%w` によるラップ

Goには例外がなく、失敗しうる関数は最後の戻り値に `error` を返すのが慣習
（`(Document, error)` のような複数戻り値）。`fmt.Errorf("claim: %w", err)` の `%w` は
元のエラーを内部に保持したまま新しいメッセージを追加する「ラップ」。呼び出し元は
`errors.Is(err, ErrNoDocument)` でラップの奥にある元のエラーと比較できる
（`store.go` の `ErrNoDocument` がその例）。

### 6. `defer`

`defer tx.Rollback()` や `defer rows.Close()` は、関数が return する直前（正常終了でも
panicでも）に必ず実行される後始末処理を登録する。PHPの `finally` やTSの
`try/finally` に近いが、関数の先頭で「後でこれをやる」と宣言できるのが特徴。
`tx.Commit()` が成功していれば、その後の `Rollback()` は無害なno-opになる。

### 7. `context.Context` によるキャンセル伝播

`ctx context.Context` を引数の先頭で受け取るのはGoの慣習。`main.go` の
`signal.NotifyContext` はSIGINT/SIGTERMを受けると `ctx.Done()` チャネルを閉じ、
その `ctx` を渡している全ての関数（DBクエリ、HTTPリクエストなど）に「中断してよい」
というシグナルを伝播できる。

### 8. チャネルと `select`

`internal/worker/worker.go` の `Run` メソッドは `time.Ticker` が2秒ごとに送る
`ticker.C` チャネルと、`ctx.Done()` チャネルを `select` で同時に待つ。先に届いた
方を処理するという、Go特有の並行処理プリミティブ。TSでいう
`Promise.race` に近い発想。

### 9. 構造体タグ (struct tag)

`internal/embedding/client.go` の `Values []float64 \`json:"values"\`` のような
バッククォート文字列が構造体タグ。`encoding/json` パッケージがこれを読んで
Go側のフィールド名とJSONのキー名（`values`）を対応付ける。TSの
`@JsonProperty` デコレータやPythonの `pydantic.Field(alias=...)` に近い役割。

### 10. blank import (`_ "package"`)

`store.go` の `_ "github.com/ncruces/go-sqlite3/driver"` は、そのパッケージの
名前空間は使わないが `init()` 関数（副作用）だけ実行させたいときに使う書き方。
ここでは「`database/sql` に `"sqlite3"` という名前のドライバを登録する」という
副作用のためだけにimportしている。

### 11. テスト (`testing` パッケージ)

Goの標準ライブラリだけでテストが書ける（`*_test.go` ファイル + `go test`）。
`t.Helper()` はヘルパー関数だと宣言してエラー行を呼び出し元に見せる、
`t.Cleanup(...)` は `defer` のテスト版、`t.TempDir()` はテスト終了時に自動削除される
一時ディレクトリを返す。`net/http/httptest.Server` は実際にlocalhostでHTTPサーバーを
立てるので、`internal/embedding` のテストではGemini APIをモックしたサーバーに対して
実際のHTTPリクエストを送っている。
