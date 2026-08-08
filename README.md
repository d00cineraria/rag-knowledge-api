# RAG Knowledge API

[![CI](https://github.com/d00cineraria/rag-knowledge-api/actions/workflows/ci.yml/badge.svg)](https://github.com/d00cineraria/rag-knowledge-api/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Markdown/PDFをアップロードすると、**出典引用付き**でQAできる**ローカル完結**のRAG APIサービス。

特徴は3つ:

1. **単一ファイルDBでゼロ運用** — SQLite + sqlite-vec（ベクトル検索）+ FTS5 trigram（日本語BM25）。DBサーバー不要、`uvicorn`起動だけで動く
2. **日本語ハイブリッド検索** — BM25とベクトル検索をRRFで融合し、オプションでbge-reranker-v2-m3によるリランク
3. **評価基盤を内蔵** — goldenデータセット26問に対して recall@k / MRR / nDCG / faithfulness を計測し、チャンク戦略や検索設定の変更を**数値で**判断できる

## アーキテクチャ

```
[Next.js chat UI]  ── SSE ──┐
                            ▼
[FastAPI]
  POST /v1/collections/{id}/documents ─→ パース → 見出し考慮チャンキング → embedding → 格納
  POST /v1/query ─→ ハイブリッド検索 → RRF融合 → (リランク) → ストリーミング回答 + 出典
                            │
        ┌───────────────────┼──────────────────────┐
        ▼                   ▼                      ▼
[SQLite (単一ファイル)]  [Gemini API]           [bge-reranker-v2-m3]
 sqlite-vec (KNN)        gemini-embedding-001    CPU / optional
 FTS5 trigram (BM25)     gemini-2.5-flash
```

## クイックスタート（Docker不要）

```bash
cp .env.example .env   # GEMINI_API_KEY を設定（Google AI Studioで無料発行可）

cd api
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --port 8000
# → http://localhost:8000/docs
```

UI（任意）:

```bash
cd ui && npm install && npm run dev
# → http://localhost:3000
```

docker compose派なら `docker compose up --build` でも同じ構成（api+ui）が起動します。

```bash
# APIだけで試す場合
curl -X POST localhost:8000/v1/collections \
  -H "Authorization: Bearer dev-local-key" -H "Content-Type: application/json" \
  -d '{"name":"docs"}'

curl -X POST localhost:8000/v1/collections/<id>/documents \
  -H "Authorization: Bearer dev-local-key" -F "file=@your.md;type=text/markdown"

curl -N -X POST localhost:8000/v1/query \
  -H "Authorization: Bearer dev-local-key" -H "Content-Type: application/json" \
  -d '{"collection_id":"<id>","question":"有給の繰越上限は？","stream":true}'
```

## 検索精度の評価

```bash
# コーパス投入 → 26問のgoldenデータセットで評価
python eval/setup_corpus.py --api-url http://localhost:8000 --api-key dev-local-key
python eval/run_eval.py    --api-url http://localhost:8000 --api-key dev-local-key \
  --collection-id <id> --retrieval-only

# 設定変更前後のA/B比較（チャンクサイズ、リランク有無など）
python eval/compare.py eval/results/<before>/report.json eval/results/<after>/report.json
```

- **検索指標**: recall@3 / recall@8 / MRR / nDCG@8（filename + heading_pathの連続部分列一致で正解判定）
- **生成指標**: faithfulness / answer_relevancy（Gemini構造化出力によるLLM-as-judge）
- `--retrieval-only` は生成APIを一切呼ばない（Gemini無料枠の日次クォータを消費しない）

### ベースライン実測（2026-08-08, retrieval-only, 22問）

| 指標 | SQLite (FTS5 trigram + sqlite-vec) | PostgreSQL (PGroonga + pgvector)※ |
|---|---|---|
| recall@3 | 1.000 | 1.000 |
| recall@8 | 1.000 | 1.000 |
| MRR | 0.977 | 1.000 |
| nDCG@8 | 0.980 | 1.000 |

※ PostgreSQL実装はタグ [`v0.1-postgres`](https://github.com/d00cineraria/rag-knowledge-api/releases/tag/v0.1-postgres) に保存（同一golden・同一チャンカーでの比較）。FTS5 trigram構成では1問のみ正解チャンクが2位に落ちる。

**注**: この時点のコーパスは4文書のモック規程で見出し構造が明瞭なため天井付近に張り付いていた（ハイブリッド検索の健全性確認としては有効だが、評価セットの識別力は不足）。Phase 2でコーパス・golden双方を強化し、下記の通り天井を剥がした。

### Phase 2: 評価セットの識別力強化（2026-08-08, retrieval-only, SQLite）

コーパスに紛らわしいモック文書を4本追加（計8文書）し、goldenに18問追加（計44問: 言い換え／数値が紛らわしい単一文書特定／複数文書横断／旧版と現行の区別／出典なしの5類型）。

| 指標 | Phase 1（4文書・26問） | Phase 2（8文書・44問） | 差分 |
|---|---|---|---|
| recall@3 | 1.000 | 0.946 | -0.054 |
| recall@8 | 1.000 | 1.000 | +0.000 |
| MRR | 0.977 | 0.752 | -0.225 |
| nDCG@8 | 0.980 | 0.813 | -0.166 |

天井は剥がれた（MRR・nDCG@8が0.7〜0.95帯に低下）。既存26問のうち複数問（例: 年次有給休暇の繰越日数、退職届出期限、情報資産の格付け）も、紛らわしい旧版・関連規程の追加によりMRRが1.0→0.5前後に低下しており、追加コーパスが意図通り識別力を要求する内容になっていることを確認した（recall@8は依然1.000で、正解チャンク自体は取得できているが上位順位付けの精度が問われる形になった）。追加文書は `eval/corpus/SOURCES.md` に自作モックである旨を明記。

## 設計上の主な判断

| 論点 | 判断 | 理由 |
|---|---|---|
| DB | SQLite単一ファイル | ローカル完結・ゼロ運用。中小規模の文書QAにはDBサーバーは過剰。PostgreSQL版（pgvector+PGroonga）は`v0.1-postgres`タグに保持し、スケール要件が出たら差し替え可能な構造 |
| 日本語全文検索 | FTS5 trigram + 軽量クエリ分解 | 形態素解析の依存を増やさずCJKの部分文字列一致でBM25を成立させる。質問文は助詞ベースでOR分解（`build_fts_query`、純粋関数でテスト済み） |
| 融合 | RRF (k=60) | スコアの正規化不要でBM25とcosineを安全に混ぜられる |
| embedding | gemini-embedding-001 (768次元・L2正規化) | 低コスト・多言語。次元はMRL truncationで768に固定 |
| リランカー | bge-reranker-v2-m3 / opt-in | CPU実行可・無料。重い依存はrequirements-reranker.txtに分離 |
| 評価 | 内蔵・CI組込 | チューニングの根拠を数値で残す（このリポジトリの主眼） |

詳細な取り決めは [docs/contracts.md](docs/contracts.md)（並列開発時のインターフェース契約）を参照。

## ロードマップ

- [x] Phase 1: Python MVP（取り込み / ハイブリッド検索 / SSE回答 / 評価基盤 / UI）
- [x] Phase 1.5: SQLite + sqlite-vec によるローカル完結化（PostgreSQL版は`v0.1-postgres`）
- [x] Phase 2: 評価セットの識別力強化（文書追加・言い換え・複数文書横断問題）※生成指標の実測はGemini生成API無料枠回復後に別途実施
- [ ] Phase 3: 取り込みワーカーのGo実装

## 開発

```bash
cd api
pip install -r requirements-dev.txt
ruff check . && pytest -q          # APIテスト43件（実SQLiteでの統合テスト含む）
python -m pytest ../eval/tests     # 評価基盤テスト30件
```

## License

MIT
