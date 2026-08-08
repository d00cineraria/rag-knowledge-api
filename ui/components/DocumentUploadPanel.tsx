"use client";

import { useMemo, useRef, useState } from "react";
import { uploadDocument } from "@/lib/api";
import { addTrackedDocument, getTrackedDocuments, type TrackedDocument } from "@/lib/storage";
import { useDocumentPolling } from "@/lib/useDocumentPolling";
import { StatusBadge } from "./StatusBadge";

export function DocumentUploadPanel({
  apiUrl,
  apiKey,
  collectionId,
}: {
  apiUrl: string;
  apiKey: string;
  collectionId: string;
}) {
  const [tracked, setTracked] = useState<TrackedDocument[]>(() =>
    getTrackedDocuments().filter((d) => d.collection_id === collectionId)
  );
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const documentIds = useMemo(() => tracked.map((d) => d.document_id), [tracked]);
  const statuses = useDocumentPolling(apiUrl, apiKey, documentIds);

  const handleFile = async (file: File) => {
    setUploading(true);
    setError(null);
    try {
      const result = await uploadDocument(apiUrl, apiKey, collectionId, file);
      const doc: TrackedDocument = {
        document_id: result.document_id,
        filename: file.name,
        collection_id: collectionId,
      };
      addTrackedDocument(doc);
      setTracked((prev) => [doc, ...prev]);
    } catch (err) {
      setError(err instanceof Error ? err.message : "アップロードに失敗しました");
    } finally {
      setUploading(false);
      if (fileInputRef.current) fileInputRef.current.value = "";
    }
  };

  return (
    <div className="flex flex-col gap-3">
      <div>
        <label className="flex cursor-pointer items-center justify-center gap-2 rounded-lg border-2 border-dashed border-neutral-300 px-4 py-6 text-sm text-neutral-500 hover:border-neutral-400 hover:bg-neutral-50">
          <input
            ref={fileInputRef}
            type="file"
            accept=".md,.pdf"
            className="hidden"
            disabled={uploading}
            onChange={(e) => {
              const file = e.target.files?.[0];
              if (file) handleFile(file);
            }}
          />
          {uploading ? "アップロード中..." : ".md / .pdf をアップロード"}
        </label>
        {error && <p className="mt-2 text-xs text-red-600">{error}</p>}
      </div>

      {tracked.length > 0 && (
        <ul className="flex flex-col gap-2">
          {tracked.map((doc) => {
            const status = statuses[doc.document_id];
            return (
              <li
                key={doc.document_id}
                className="flex items-center justify-between rounded-md border border-neutral-200 px-3 py-2 text-sm"
              >
                <span className="truncate text-neutral-800">{doc.filename}</span>
                <div className="flex items-center gap-2">
                  {status?.status === "error" && status.error && (
                    <span className="max-w-40 truncate text-xs text-red-600" title={status.error}>
                      {status.error}
                    </span>
                  )}
                  <StatusBadge status={status?.status ?? "pending"} />
                </div>
              </li>
            );
          })}
        </ul>
      )}
    </div>
  );
}
