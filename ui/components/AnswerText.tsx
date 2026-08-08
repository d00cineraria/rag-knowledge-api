const CITATION_RE = /(\[\d+\])/g;

export function AnswerText({ text, sourceIndexes }: { text: string; sourceIndexes: Set<number> }) {
  const parts = text.split(CITATION_RE);

  return (
    <p className="whitespace-pre-wrap text-sm leading-relaxed text-neutral-800">
      {parts.map((part, i) => {
        const match = part.match(/^\[(\d+)\]$/);
        const index = match ? Number(match[1]) : null;
        if (index !== null && sourceIndexes.has(index)) {
          return (
            <a
              key={i}
              href={`#source-${index}`}
              className="mx-0.5 inline-flex h-4 min-w-4 items-center justify-center rounded bg-neutral-200 px-1 align-super text-[10px] font-semibold text-neutral-700 no-underline hover:bg-neutral-300"
            >
              {index}
            </a>
          );
        }
        return <span key={i}>{part}</span>;
      })}
    </p>
  );
}
