"use client";

import { useState } from "react";
import { useAppConfig } from "@/app/providers";

export function SettingsDialog({ onClose }: { onClose: () => void }) {
  const { apiKey, apiUrl, setApiKey, setApiUrl } = useAppConfig();
  const [keyDraft, setKeyDraft] = useState(apiKey);
  const [urlDraft, setUrlDraft] = useState(apiUrl);

  const save = () => {
    setApiKey(keyDraft.trim());
    setApiUrl(urlDraft.trim() || apiUrl);
    onClose();
  };

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/30 px-4"
      onClick={onClose}
    >
      <div
        className="w-full max-w-md rounded-lg border border-neutral-200 bg-white p-6 shadow-xl"
        onClick={(e) => e.stopPropagation()}
      >
        <h2 className="text-base font-semibold text-neutral-900">設定</h2>
        <p className="mt-1 text-sm text-neutral-500">
          APIキーとエンドポイントはブラウザのlocalStorageに保存されます。
        </p>

        <div className="mt-5 flex flex-col gap-4">
          <label className="flex flex-col gap-1.5 text-sm">
            <span className="font-medium text-neutral-700">API URL</span>
            <input
              type="text"
              value={urlDraft}
              onChange={(e) => setUrlDraft(e.target.value)}
              placeholder="http://localhost:8000"
              className="rounded-md border border-neutral-300 px-3 py-2 text-sm outline-none focus:border-neutral-500"
            />
          </label>

          <label className="flex flex-col gap-1.5 text-sm">
            <span className="font-medium text-neutral-700">API Key</span>
            <input
              type="password"
              value={keyDraft}
              onChange={(e) => setKeyDraft(e.target.value)}
              placeholder="dev-local-key"
              className="rounded-md border border-neutral-300 px-3 py-2 text-sm outline-none focus:border-neutral-500"
            />
          </label>
        </div>

        <div className="mt-6 flex justify-end gap-2">
          <button
            onClick={onClose}
            className="rounded-md px-3 py-1.5 text-sm text-neutral-600 hover:bg-neutral-100"
          >
            キャンセル
          </button>
          <button
            onClick={save}
            className="rounded-md bg-neutral-900 px-3 py-1.5 text-sm font-medium text-white hover:bg-neutral-700"
          >
            保存
          </button>
        </div>
      </div>
    </div>
  );
}
