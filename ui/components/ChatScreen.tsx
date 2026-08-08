"use client";

import { useRef, useState, type FormEvent } from "react";
import Link from "next/link";
import { useAppConfig } from "@/app/providers";
import { openQueryStream } from "@/lib/api";
import { parseSSEStream } from "@/lib/sse";
import type { LatencyMs, Source } from "@/lib/types";
import { SourceCard } from "./SourceCard";
import { AnswerText } from "./AnswerText";

type ChatMessage = {
  id: string;
  question: string;
  sources: Source[];
  answer: string;
  latency: LatencyMs | null;
  status: "streaming" | "done" | "error";
  error?: string;
};

let messageCounter = 0;

export function ChatScreen() {
  const { apiUrl, apiKey, selectedCollectionId, selectedCollectionName, hydrated } =
    useAppConfig();
  const [question, setQuestion] = useState("");
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [busy, setBusy] = useState(false);
  const abortRef = useRef<AbortController | null>(null);

  const updateMessage = (
    id: string,
    patch: Partial<ChatMessage> | ((m: ChatMessage) => Partial<ChatMessage>)
  ) => {
    setMessages((prev) =>
      prev.map((m) =>
        m.id === id ? { ...m, ...(typeof patch === "function" ? patch(m) : patch) } : m
      )
    );
  };

  const submit = async (e: FormEvent) => {
    e.preventDefault();
    const q = question.trim();
    if (!q || !selectedCollectionId || busy) return;

    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;

    const id = `m${++messageCounter}`;
    const message: ChatMessage = {
      id,
      question: q,
      sources: [],
      answer: "",
      latency: null,
      status: "streaming",
    };
    setMessages((prev) => [...prev, message]);
    setQuestion("");
    setBusy(true);

    try {
      const res = await openQueryStream(
        apiUrl,
        apiKey,
        selectedCollectionId,
        q,
        8,
        controller.signal
      );
      for await (const evt of parseSSEStream(res.body!)) {
        if (evt.event === "sources") {
          const parsed = JSON.parse(evt.data) as { sources: Source[] };
          updateMessage(id, { sources: parsed.sources });
        } else if (evt.event === "token") {
          const parsed = JSON.parse(evt.data) as { text: string };
          updateMessage(id, (m) => ({ answer: m.answer + parsed.text }));
        } else if (evt.event === "done") {
          const parsed = JSON.parse(evt.data) as { latency_ms: LatencyMs };
          updateMessage(id, { latency: parsed.latency_ms, status: "done" });
        }
      }
      updateMessage(id, (m) => (m.status === "streaming" ? { status: "done" } : {}));
    } catch (err) {
      if (controller.signal.aborted) return;
      updateMessage(id, {
        status: "error",
        error: err instanceof Error ? err.message : "エラーが発生しました",
      });
    } finally {
      setBusy(false);
    }
  };

  if (!hydrated) return null;

  if (!apiKey) {
    return (
      <div className="rounded-lg border border-dashed border-neutral-300 p-8 text-center text-sm text-neutral-500">
        右上の「設定」からAPIキーを入力してください。
      </div>
    );
  }

  if (!selectedCollectionId) {
    return (
      <div className="rounded-lg border border-dashed border-neutral-300 p-8 text-center text-sm text-neutral-500">
        質問するには先に
        <Link href="/" className="mx-1 font-medium text-neutral-800 underline">
          コレクションを選択
        </Link>
        してください。
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-6">
      <p className="text-xs text-neutral-500">
        対象コレクション:{" "}
        <span className="font-medium text-neutral-700">{selectedCollectionName}</span>
      </p>

      <div className="flex flex-col gap-6">
        {messages.length === 0 && (
          <p className="text-sm text-neutral-400">質問を入力して送信してください。</p>
        )}
        {messages.map((m) => (
          <ChatMessageView key={m.id} message={m} />
        ))}
      </div>

      <form onSubmit={submit} className="sticky bottom-4 flex gap-2">
        <input
          type="text"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder="質問を入力..."
          maxLength={2000}
          className="flex-1 rounded-lg border border-neutral-300 bg-white px-4 py-2.5 text-sm shadow-sm outline-none focus:border-neutral-500"
        />
        <button
          type="submit"
          disabled={busy || !question.trim()}
          className="rounded-lg bg-neutral-900 px-4 py-2.5 text-sm font-medium text-white hover:bg-neutral-700 disabled:cursor-not-allowed disabled:opacity-40"
        >
          {busy ? "送信中..." : "送信"}
        </button>
      </form>
    </div>
  );
}

function ChatMessageView({ message }: { message: ChatMessage }) {
  const sourceIndexes = new Set(message.sources.map((s) => s.index));

  return (
    <div className="flex flex-col gap-3">
      <div className="max-w-[85%] self-end rounded-2xl rounded-br-sm bg-neutral-900 px-4 py-2 text-sm text-white">
        {message.question}
      </div>

      {message.sources.length > 0 && (
        <div className="grid grid-cols-1 gap-2 sm:grid-cols-2">
          {message.sources.map((s) => (
            <SourceCard key={s.chunk_id} source={s} />
          ))}
        </div>
      )}

      <div className="rounded-2xl rounded-bl-sm border border-neutral-200 bg-white px-4 py-3">
        {message.answer ? (
          <AnswerText text={message.answer} sourceIndexes={sourceIndexes} />
        ) : message.status === "streaming" ? (
          <p className="text-sm text-neutral-400">回答を生成中...</p>
        ) : null}

        {message.status === "error" && (
          <p className="mt-1 text-xs text-red-600">{message.error}</p>
        )}

        {message.latency && (
          <p className="mt-2 text-[11px] text-neutral-400">
            検索 {message.latency.retrieval}ms ・ 生成 {message.latency.generation}ms
          </p>
        )}
      </div>
    </div>
  );
}
