# インターフェース契約（WS0で固定 — 変更にはオーケストレーターの承認が必要）

並列ワークストリーム（WS1〜WS4）はこの契約に従って実装する。契約の変更が必要になった場合は、勝手に変えずにオーケストレーターへ報告すること。

## 担当範囲（ファイル所有権 — 他WSの領域を編集しない）

| WS | ブランチ | 所有ディレクトリ/ファイル |
|---|---|---|
| WS1 取り込み | `ws1-ingest` | `api/app/services/ingest/**`, `api/tests/test_ingest*.py` |
| WS2 検索・回答 | `ws2-query` | `api/app/services/retrieval/**`, `api/app/services/generation/**`, `api/tests/test_retrieval*.py`, `api/tests/test_generation*.py` |
| WS3 評価基盤 | `ws3-eval` | `eval/**`, `.github/workflows/eval.yml` |
| WS4 UI | `ws4-ui` | `ui/**`（docker-compose.ymlへのuiサービス追加はマージ時にオーケストレーターが行う） |

共有ファイル（routers/, schemas.py, db.py, config.py, docker-compose.yml, 001_schema.sql）は原則WS0のまま。変更が必要なら報告。

## DBスキーマ

`db/init/001_schema.sql` が正本。要点:
- `chunks.embedding` は **vector(768)**（Gemini `gemini-embedding-001` を output_dimensionality=768 で使用、**正規化してから格納**）
- 日本語全文検索は PGroonga（`&@~` 演算子 + `pgroonga_score()`）
- ベクトル検索は pgvector cosine（`<=>` 演算子）

## サービス層インターフェース（api/app/services/）

ルーター（WS0所有）はこの関数シグネチャを呼ぶ。実装本体を各WSが埋める。

```python
# services/ingest/__init__.py  (WS1が実装)
async def process_document(document_id: UUID) -> None:
    """documentsレコード(status=pending)を読み、/data/raw/{document_id} のファイルを
    パース→チャンキング→embedding→chunksへINSERT→status=ready に更新。
    失敗時は status=error, error=メッセージ。FastAPI BackgroundTasksから呼ばれる。"""

# services/retrieval/__init__.py  (WS2が実装)
async def search(collection_id: UUID, question: str, top_k: int = 8) -> list[RetrievedChunk]:
    """PGroonga BM25 + pgvector cosine を各候補N=30件取得し RRF(k=60) で融合、
    RERANKER_ENABLED時は bge-reranker-v2-m3 で上位をリランクして top_k 件返す。"""

# services/generation/__init__.py  (WS2が実装)
async def stream_answer(question: str, chunks: list[RetrievedChunk]) -> AsyncIterator[str]:
    """Geminiで出典引用付き回答をトークン単位でyield。出典は [1][2] 形式で本文中に引用。"""
```

`RetrievedChunk` は `api/app/schemas.py` に定義済み（chunk_id, document_id, filename, heading_path, content, score）。

## API仕様（確定）

認証: `Authorization: Bearer <key>`。開発キーは `.env` の `API_DEV_KEY`（起動時に自動シード）。

| Method | Path | 説明 |
|---|---|---|
| GET | /healthz | 認証不要。`{"status":"ok"}` |
| POST | /v1/collections | `{name, description?}` → 201 collection |
| GET | /v1/collections | 一覧 |
| POST | /v1/collections/{collection_id}/documents | multipart `file`（.md/.pdf）→ 202 `{document_id, status:"pending"}`。BackgroundTasksで`process_document`起動 |
| GET | /v1/documents/{document_id} | `{id, filename, status, error?}` |
| POST | /v1/query | 下記 |

### POST /v1/query

リクエスト: `{"collection_id": "...", "question": "...", "top_k": 8, "stream": true}`

stream=true → SSE:
```
event: sources
data: {"sources": [{"index":1,"chunk_id":"...","filename":"...","heading_path":[...],"content":"...","score":0.87}, ...]}

event: token
data: {"text": "回答の断片"}

event: done
data: {"latency_ms": {"retrieval": 120, "generation": 900}}
```
stream=false → JSON `{"answer": "...", "sources": [...]}`

## 評価データ形式（WS3）

`eval/golden/golden.jsonl` — 1行1問:
```json
{"id": "q001", "question": "...", "reference_answer": "...", "relevant": [{"filename": "doc.md", "heading_path": ["第2章"]}]}
```
- 検索指標: recall@k (k=3,8), MRR, nDCG@8（relevantとchunkの突合はfilename一致 + heading_pathの**連続部分列一致**。チャンカーはheading_path先頭にH1タイトルを含むため、先頭固定の前方一致は使わない — 2026-08-08統合時に改定）
- 生成指標: faithfulness / answer_relevancy（LLM-as-judge、Gemini使用）
- 実行: `python eval/run_eval.py --api-url http://localhost:8000 --api-key ...` → `eval/results/` にJSON+Markdownレポート
- サンプル文書は**公開ライセンスのもののみ**（省庁ガイドライン等）。`eval/corpus/` に置き出典をREADMEに明記

## UI（WS4）

- `ui/` に Next.js (App Router, TypeScript)。`NEXT_PUBLIC_API_URL` でAPIを指す
- 画面: ①コレクション選択+文書アップロード ②チャット（SSE受信、`sources`イベントを出典カードとして表示、`token`を逐次描画）
- APIキーは画面の設定欄で入力しlocalStorageに保持（デモ用途）

## 環境変数

`.env.example` が正本。追加が必要なら報告してから追加。
