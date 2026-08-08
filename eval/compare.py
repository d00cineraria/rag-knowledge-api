"""2つの評価レポート(report.json)を比較し、指標差分のMarkdownテーブルを出す。

チャンク戦略のA/B比較などに使う小さなツール。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

_METRIC_KEYS = ["recall@3", "recall@8", "mrr", "ndcg@8", "faithfulness", "answer_relevancy"]

_METRIC_LABELS = {
    "recall@3": "recall@3",
    "recall@8": "recall@8",
    "mrr": "MRR",
    "ndcg@8": "nDCG@8",
    "faithfulness": "faithfulness (1-5)",
    "answer_relevancy": "answer_relevancy (1-5)",
}


def load_report(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def build_diff_table(
    report_a: dict[str, Any], report_b: dict[str, Any], *, label_a: str = "A", label_b: str = "B"
) -> str:
    agg_a = report_a["aggregate"]
    agg_b = report_b["aggregate"]
    lines = [
        "# A/B比較レポート",
        "",
        f"| 指標 | {label_a} | {label_b} | 差分 (B-A) |",
        "|---|---|---|---|",
    ]
    for key in _METRIC_KEYS:
        va = agg_a.get(key)
        vb = agg_b.get(key)
        if va is None or vb is None:
            fa = f"{va:.3f}" if va is not None else "-"
            fb = f"{vb:.3f}" if vb is not None else "-"
            lines.append(f"| {_METRIC_LABELS[key]} | {fa} | {fb} | - |")
            continue
        diff = vb - va
        sign = "+" if diff >= 0 else ""
        lines.append(f"| {_METRIC_LABELS[key]} | {va:.3f} | {vb:.3f} | {sign}{diff:.3f} |")
    return "\n".join(lines) + "\n"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="2つのreport.jsonの指標差分を比較する")
    parser.add_argument("report_a", type=Path)
    parser.add_argument("report_b", type=Path)
    parser.add_argument("--label-a", default="A")
    parser.add_argument("--label-b", default="B")
    parser.add_argument("--out", type=Path, default=None, help="未指定時は標準出力")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    table = build_diff_table(
        load_report(args.report_a),
        load_report(args.report_b),
        label_a=args.label_a,
        label_b=args.label_b,
    )
    if args.out:
        args.out.write_text(table, encoding="utf-8")
    else:
        print(table)


if __name__ == "__main__":
    main()
