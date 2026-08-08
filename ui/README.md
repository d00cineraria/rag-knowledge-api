# RAG Portfolio UI

RAG Knowledge API の最小フロントエンド（Next.js App Router + TypeScript + Tailwind CSS）。

## 画面構成

- `/` — コレクション一覧・作成、`.md` / `.pdf` アップロード（ステータスを2秒間隔でポーリングし `pending → ready/error` を表示）
- `/chat` — 質問送信 → `POST /v1/query`（`stream: true`）→ `fetch` + `ReadableStream` でSSEを受信
  - `sources` イベント: 出典カード（filename・heading_path・score・本文の折りたたみ表示）
  - `token` イベント: 回答テキストを逐次描画。本文中の `[1]` `[2]` は対応する出典カードへのアンカーリンクになる
  - `done` イベント: 検索/生成レイテンシを表示

設定（右上の「設定」ボタン）でAPI URLとAPIキーを入力し、`localStorage` に保持する。

## セットアップ

```bash
npm install
cp .env.local.example .env.local   # NEXT_PUBLIC_API_URL を必要に応じて変更
npm run dev
```

`NEXT_PUBLIC_API_URL` 未設定時は `http://localhost:8000` を使用する。

## ビルド

```bash
npm run build
```

このリポジトリの実行環境では Turbopack がローカルポートへのバインドに失敗するため、`build`/`dev` スクリプトは `--webpack` を指定している。

## 実装メモ

- SSEはPOSTで受けるため `EventSource` は使えず、`fetch` のレスポンスボディ（`ReadableStream`）を `lib/sse.ts` の `parseSSEStream` で手動パースしている
- APIレスポンス型は `api/app/schemas.py` に合わせて `lib/types.ts` に定義
- アップロード済みドキュメントの一覧はサーバー側に一覧APIが無いため、`localStorage` にクライアント側で追跡している（`lib/storage.ts`）
