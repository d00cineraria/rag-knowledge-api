import { useEffect, useRef, useState } from "react";
import { getDocumentStatus } from "./api";
import type { DocumentStatus } from "./types";

const POLL_INTERVAL_MS = 2000;

/** 指定ドキュメントIDのステータスをpending中のみ定期ポーリングする。 */
export function useDocumentPolling(
  apiUrl: string,
  apiKey: string,
  documentIds: string[]
): Record<string, DocumentStatus> {
  const [statuses, setStatuses] = useState<Record<string, DocumentStatus>>({});
  const statusesRef = useRef(statuses);
  statusesRef.current = statuses;

  useEffect(() => {
    if (documentIds.length === 0 || !apiKey) return;

    let cancelled = false;

    const poll = async () => {
      const pending = documentIds.filter((id) => {
        const current = statusesRef.current[id];
        return !current || current.status === "pending";
      });
      if (pending.length === 0) return;

      const results = await Promise.all(
        pending.map(async (id) => {
          try {
            return await getDocumentStatus(apiUrl, apiKey, id);
          } catch {
            return null;
          }
        })
      );

      if (cancelled) return;
      setStatuses((prev) => {
        const next = { ...prev };
        for (const result of results) {
          if (result) next[result.id] = result;
        }
        return next;
      });
    };

    poll();
    const interval = setInterval(poll, POLL_INTERVAL_MS);
    return () => {
      cancelled = true;
      clearInterval(interval);
    };
  }, [apiUrl, apiKey, documentIds.join(",")]);

  return statuses;
}
