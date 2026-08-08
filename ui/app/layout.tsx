import type { Metadata } from "next";
import "./globals.css";
import { AppConfigProvider } from "./providers";
import { Header } from "@/components/Header";

export const metadata: Metadata = {
  title: "RAG Portfolio",
  description: "コレクション管理とチャットのための最小UI",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html lang="ja" className="h-full antialiased">
      <body className="flex min-h-full flex-col bg-white font-sans text-neutral-900">
        <AppConfigProvider>
          <Header />
          <main className="mx-auto w-full max-w-4xl flex-1 px-6 py-8">{children}</main>
        </AppConfigProvider>
      </body>
    </html>
  );
}
