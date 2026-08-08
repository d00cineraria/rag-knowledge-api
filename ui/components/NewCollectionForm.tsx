"use client";

import { useState, type FormEvent } from "react";
import { createCollection } from "@/lib/api";
import type { Collection } from "@/lib/types";

export function NewCollectionForm({
  apiUrl,
  apiKey,
  onCreated,
}: {
  apiUrl: string;
  apiKey: string;
  onCreated: (collection: Collection) => void;
}) {
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const submit = async (e: FormEvent) => {
    e.preventDefault();
    if (!name.trim()) return;
    setSubmitting(true);
    setError(null);
    try {
      const collection = await createCollection(apiUrl, apiKey, name.trim(), description.trim());
      setName("");
      setDescription("");
      onCreated(collection);
    } catch (err) {
      setError(err instanceof Error ? err.message : "作成に失敗しました");
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <form onSubmit={submit} className="flex flex-col gap-2 rounded-lg border border-neutral-200 p-4">
      <h3 className="text-sm font-semibold text-neutral-900">新規コレクション</h3>
      <input
        type="text"
        value={name}
        onChange={(e) => setName(e.target.value)}
        placeholder="名前"
        maxLength={100}
        className="rounded-md border border-neutral-300 px-3 py-1.5 text-sm outline-none focus:border-neutral-500"
      />
      <textarea
        value={description}
        onChange={(e) => setDescription(e.target.value)}
        placeholder="説明（任意）"
        rows={2}
        className="resize-none rounded-md border border-neutral-300 px-3 py-1.5 text-sm outline-none focus:border-neutral-500"
      />
      {error && <p className="text-xs text-red-600">{error}</p>}
      <button
        type="submit"
        disabled={submitting || !name.trim()}
        className="mt-1 rounded-md bg-neutral-900 px-3 py-1.5 text-sm font-medium text-white hover:bg-neutral-700 disabled:cursor-not-allowed disabled:opacity-40"
      >
        {submitting ? "作成中..." : "作成"}
      </button>
    </form>
  );
}
