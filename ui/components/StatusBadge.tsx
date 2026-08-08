const STYLES: Record<string, string> = {
  pending: "bg-amber-50 text-amber-700 border-amber-200",
  ready: "bg-emerald-50 text-emerald-700 border-emerald-200",
  error: "bg-red-50 text-red-700 border-red-200",
};

const LABELS: Record<string, string> = {
  pending: "処理中",
  ready: "準備完了",
  error: "エラー",
};

export function StatusBadge({ status }: { status: string }) {
  const style = STYLES[status] ?? "bg-neutral-50 text-neutral-600 border-neutral-200";
  const label = LABELS[status] ?? status;
  return (
    <span className={`inline-flex items-center gap-1 rounded-full border px-2 py-0.5 text-xs font-medium ${style}`}>
      {status === "pending" && (
        <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-amber-500" />
      )}
      {label}
    </span>
  );
}
