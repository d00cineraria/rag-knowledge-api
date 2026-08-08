# RAG Knowledge API

[![CI](https://github.com/d00cineraria/rag-knowledge-api/actions/workflows/ci.yml/badge.svg)](https://github.com/d00cineraria/rag-knowledge-api/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Markdown/PDFをアップロードすると、**出典引用付き**でQAできるRAG APIサービス。

特徴は2つ:

1. **日本語ハイブリッド検索** — PGroonga（BM25）+ pgvector（cosine/HNSW）をRRFで融合し、オプションでbge-reranker-v2-m3によるリランク
2. **評価基盤を内蔵** — goldenデータセット26問に対して recall@k / MRR / nDCG / faithfulness を計測し、チャンク戦略や検索設定の変更を**数値で**判断できる（「なんとなく良くなった」で終わらせない）

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
[PostgreSQL 16]       [Gemini API]           [bge-reranker-v2-m3]
 pgvector (HNSW)       gemini-embedding-001    CPU / optional
 PGroonga (日本語BM25)  gemini-2.5-flash
```

## クイックスタート

```bash
cp .env.example .env   # GEMINI_API_KEY を設定（Google AI Studioで無料発行可）
docker compose up --build
```

| URL | 内容 |
|---|---|
| http://localhost:3000 | チャットUI（コレクション管理・アップロード・SSEチャット） |
| http://localhost:8000/docs | OpenAPI (Swagger UI) |

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
python eval/run_eval.py    --api-url http://localhost:8000 --api-key dev-local-key

# 設定変更前後のA/B比較（チャンクサイズ、リランク有無など）
python eval/compare.py eval/results/<before>/report.json eval/results/<after>/report.json
```

- **検索指標**: recall@3 / recall@8 / MRR / nDCG@8（filename + heading_pathの連続部分列一致で正解判定）
- **生成指標**: faithfulness / answer_relevancy（Gemini構造化出力によるLLM-as-judge）
- `--retrieval-only` で生成APIを一切呼ばずに検索指標のみ計測可能（無料枠の日次クォータを消費しない）
- GitHub Actions（workflow_dispatch）でスタック起動→評価→レポートartifact保存まで自動実行

### ベースライン実測（2026-08-08, retrieval-only）

| 指標 | 値 |
|---|---|
| recall@3 | 1.000 |
| recall@8 | 1.000 |
| MRR | 1.000 |
| nDCG@8 | 1.000 |

対象22問（+出典なし想定4問）。**注**: 現行コーパスは4文書のモック規程で見出し構造が明瞭なため天井に張り付いている（ハイブリッド検索の健全性確認としては有効だが、評価セットの識別力は不足）。次ステップとして、文書数を増やし、言い換え・複数文書横断・数値の紛らわしい問題を足して天井を剥がす予定。

## 設計上の主な判断

| 論点 | 判断 | 理由 |
|---|---|---|
| 日本語全文検索 | PGroonga | tsvectorは日本語トークナイズが弱い。運用DBを増やさずBM25相当を得る |
| 融合 | RRF (k=60) | スコアの正規化不要でBM25とcosineを安全に混ぜられる |
| embedding | gemini-embedding-001 (768次元・L2正規化) | 低コスト・多言語。次元はMRL truncationで768に固定 |
| リランカー | bge-reranker-v2-m3 / opt-in | CPU実行可・無料。重い依存はrequirements-reranker.txtに分離 |
| 評価 | 内蔵・CI組込 | チューニングの根拠を数値で残す（このリポジトリの主眼） |

詳細な取り決めは [docs/contracts.md](docs/contracts.md)（並列開発時のインターフェース契約）を参照。

## ロードマップ

- [x] Phase 1: Python MVP（取り込み / ハイブリッド検索 / SSE回答 / 評価基盤 / UI）
- [ ] Phase 1.5: Cloud Runデモ公開
- [ ] Phase 2: 取り込みワーカーのGo実装
- [ ] Phase 3: Terraform + AWSデプロイ（構成証跡）
- [ ] Phase 3.5: Cloudflare Workers AI + Vectorize ライト構成の設計ドキュメント

## 開発

```bash
cd api
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
ruff check . && pytest -q        # APIユニットテスト38件
python -m pytest ../eval/tests   # 評価基盤テスト30件
```

## License

MIT
