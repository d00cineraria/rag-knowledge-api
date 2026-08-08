"use client";

import { useCallback, useEffect, useState } from "react";
import { useAppConfig } from "@/app/providers";
import { listCollections } from "@/lib/api";
import type { Collection } from "@/lib/types";
import { NewCollectionForm } from "./NewCollectionForm";
import { DocumentUploadPanel } from "./DocumentUploadPanel";

export function CollectionsScreen() {
  const { apiUrl, apiKey, selectedCollectionId, setSelectedCollection, hydrated } = useAppConfig();
  const [collections, setCollections] = useState<Collection[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const refresh = useCallback(async () => {
    if (!apiKey) {
      setCollections(null);
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const data = await listCollections(apiUrl, apiKey);
      setCollections(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "コレクションの取得に失敗しました");
    } finally {
      setLoading(false);
    }
  }, [apiUrl, apiKey]);

  useEffect(() => {
    if (!hydrated) return;
    refresh();
  }, [hydrated, refresh]);

  const selected = collections?.find((c) => c.id === selectedCollectionId) ?? null;

  if (!hydrated) return null;

  if (!apiKey) {
    return (
      <div className="rounded-lg border border-dashed border-neutral-300 p-8 text-center text-sm text-neutral-500">
        右上の「設定」からAPIキーを入力してください。
      </div>
    );
  }

  return (
    <div className="grid grid-cols-1 gap-8 md:grid-cols-[minmax(0,1fr)_320px]">
      <div className="flex flex-col gap-4">
        <div className="flex items-center justify-between">
          <h2 className="text-sm font-semibold text-neutral-900">コレクション一覧</h2>
          <button
            onClick={refresh}
            disabled={loading}
            className="text-xs text-neutral-500 hover:text-neutral-800 disabled:opacity-40"
          >
            {loading ? "更新中..." : "更新"}
          </button>
        </div>

        {error && (
          <p className="rounded-md border border-red-200 bg-red-50 px-3 py-2 text-xs text-red-600">
            {error}
          </p>
        )}

        {collections === null && !error && (
          <p className="text-sm text-neutral-400">読み込み中...</p>
        )}

        {collections !== null && collections.length === 0 && (
          <p className="text-sm text-neutral-400">コレクションがありません。右のフォームから作成してください。</p>
        )}

        <ul className="flex flex-col gap-2">
          {collections?.map((c) => {
            const isSelected = c.id === selectedCollectionId;
            return (
              <li key={c.id}>
                <button
                  onClick={() => setSelectedCollection(c.id, c.name)}
                  className={`w-full rounded-lg border px-4 py-3 text-left transition-colors ${
                    isSelected
                      ? "border-neutral-900 bg-neutral-50"
                      : "border-neutral-200 hover:border-neutral-400"
                  }`}
                >
                  <div className="flex items-center justify-between">
                    <span className="text-sm font-medium text-neutral-900">{c.name}</span>
                    {isSelected && (
                      <span className="text-xs font-medium text-neutral-500">選択中</span>
                    )}
                  </div>
                  {c.description && (
                    <p className="mt-1 text-xs text-neutral-500">{c.description}</p>
                  )}
                </button>
              </li>
            );
          })}
        </ul>

        {selected && (
          <div className="mt-2 flex flex-col gap-3 rounded-lg border border-neutral-200 p-4">
            <h3 className="text-sm font-semibold text-neutral-900">文書アップロード — {selected.name}</h3>
            <DocumentUploadPanel apiUrl={apiUrl} apiKey={apiKey} collectionId={selected.id} />
          </div>
        )}
      </div>

      <div>
        <NewCollectionForm
          apiUrl={apiUrl}
          apiKey={apiKey}
          onCreated={(c) => {
            setCollections((prev) => [...(prev ?? []), c]);
            setSelectedCollection(c.id, c.name);
          }}
        />
      </div>
    </div>
  );
}
