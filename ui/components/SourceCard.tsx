"use client";

import { useState } from "react";
import type { Source } from "@/lib/types";

export function SourceCard({ source }: { source: Source }) {
  const [expanded, setExpanded] = useState(false);

  return (
    <div
      id={`source-${source.index}`}
      className="scroll-mt-24 rounded-lg border border-neutral-200 bg-neutral-50 p-3 text-sm"
    >
      <div className="flex items-start justify-between gap-2">
        <div className="min-w-0">
          <div className="flex items-center gap-1.5">
            <span className="flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-neutral-900 text-[11px] font-semibold text-white">
              {source.index}
            </span>
            <span className="truncate font-medium text-neutral-900">{source.filename}</span>
          </div>
          {source.heading_path.length > 0 && (
            <p className="mt-1 truncate text-xs text-neutral-500">
              {source.heading_path.join(" > ")}
            </p>
          )}
        </div>
        <span className="shrink-0 rounded bg-white px-1.5 py-0.5 text-[11px] font-mono text-neutral-500">
          {source.score.toFixed(2)}
        </span>
      </div>

      <button
        onClick={() => setExpanded((v) => !v)}
        className="mt-2 text-xs font-medium text-neutral-500 hover:text-neutral-800"
      >
        {expanded ? "本文を隠す ▲" : "本文を表示 ▼"}
      </button>
      {expanded && (
        <p className="mt-2 whitespace-pre-wrap text-xs leading-relaxed text-neutral-700">
          {source.content}
        </p>
      )}
    </div>
  );
}
