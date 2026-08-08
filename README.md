# RAG Knowledge API

Markdown/PDFをアップロードすると、**出典引用付き**でQAできるRAG APIサービス。

日本語ハイブリッド検索（PGroonga BM25 + pgvector）と、**検索精度の評価基盤を内蔵**していることが特徴です。recall@k / MRR / nDCG / faithfulness をCIで継続計測し、チャンク戦略のA/B比較を数値で判断します。

> 🚧 開発中（Phase 1: Python MVP）。ロードマップは下記参照。

## アーキテクチャ

```
[Next.js chat UI]
      ↓
[FastAPI]
  POST /v1/collections/{id}/documents  … MD/PDF → 見出し考慮チャンキング → embedding
  POST /v1/query                       … ハイブリッド検索 → リランク → SSE回答+出典
      ↓
[PostgreSQL]  pgvector(cosine/HNSW) + PGroonga(日本語BM25) → RRF融合
[Gemini API]  gemini-embedding-001 (768dim) + gemini-2.5-flash
[Reranker]    bge-reranker-v2-m3 (CPU, optional)
```

## クイックスタート

```bash
cp .env.example .env   # GEMINI_API_KEY を設定
docker compose up --build
# API: http://localhost:8000/docs
```

## ロードマップ

- [ ] Phase 1: Python MVP（取り込み/ハイブリッド検索/SSE回答/評価基盤/UI）+ Cloud Runデモ
- [ ] Phase 2: 取り込みワーカーのGo実装
- [ ] Phase 3: Terraform + AWSデプロイ（構成証跡）
- [ ] Phase 3.5: Cloudflare Workers AI + Vectorize ライト構成の設計ドキュメント

## 開発

```bash
cd api
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
ruff check . && pytest -q
```

設計上の取り決めは [docs/contracts.md](docs/contracts.md) を参照。

## License

MIT
