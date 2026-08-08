"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useState } from "react";
import { useAppConfig } from "@/app/providers";
import { SettingsDialog } from "./SettingsDialog";

const NAV_ITEMS = [
  { href: "/", label: "コレクション" },
  { href: "/chat", label: "チャット" },
];

export function Header() {
  const pathname = usePathname();
  const { apiKey, selectedCollectionName, hydrated } = useAppConfig();
  const [settingsOpen, setSettingsOpen] = useState(false);

  return (
    <header className="border-b border-neutral-200 bg-white">
      <div className="mx-auto flex h-14 max-w-4xl items-center justify-between px-6">
        <div className="flex items-center gap-6">
          <span className="text-sm font-semibold tracking-tight text-neutral-900">RAG Portfolio</span>
          <nav className="flex items-center gap-4">
            {NAV_ITEMS.map((item) => (
              <Link
                key={item.href}
                href={item.href}
                className={`text-sm transition-colors ${
                  pathname === item.href
                    ? "font-medium text-neutral-900"
                    : "text-neutral-500 hover:text-neutral-800"
                }`}
              >
                {item.label}
              </Link>
            ))}
          </nav>
        </div>

        <div className="flex items-center gap-3">
          {hydrated && selectedCollectionName && (
            <span className="hidden text-xs text-neutral-500 sm:inline">
              選択中: <span className="font-medium text-neutral-700">{selectedCollectionName}</span>
            </span>
          )}
          {hydrated && !apiKey && (
            <span className="text-xs text-amber-600">APIキー未設定</span>
          )}
          <button
            onClick={() => setSettingsOpen(true)}
            className="rounded-md border border-neutral-300 px-3 py-1.5 text-xs font-medium text-neutral-700 hover:bg-neutral-100"
          >
            設定
          </button>
        </div>
      </div>

      {settingsOpen && <SettingsDialog onClose={() => setSettingsOpen(false)} />}
    </header>
  );
}
