#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def to_float(row: dict[str, str], key: str) -> float:
    return float(row[key])


def format_ci(mean: float, low: float, high: float, precision: int = 3) -> str:
    return f"{mean:.{precision}f} [{low:.{precision}f}, {high:.{precision}f}]"


def key_individual(row: dict[str, str]) -> tuple[str, str]:
    return row["concept_safe"], row["condition"]


def key_group(row: dict[str, str]) -> tuple[str, str]:
    return row["group"], row["condition"]


def merge_rows(
    left_rows: list[dict[str, str]],
    right_rows: list[dict[str, str]],
    key_fn,
    right_prefix: str,
    left_prefix: str,
) -> list[dict[str, object]]:
    right_index = {key_fn(row): row for row in right_rows}
    merged: list[dict[str, object]] = []
    for left_row in left_rows:
        key = key_fn(left_row)
        right_row = right_index.get(key)
        if right_row is None:
            continue
        row: dict[str, object] = {}
        for field, value in left_row.items():
            row[f"{left_prefix}{field}"] = value
        for field, value in right_row.items():
            row[f"{right_prefix}{field}"] = value
        merged.append(row)
    return merged


def build_individual_comparison(
    dreambooth_rows: list[dict[str, str]],
    baseline_rows: list[dict[str, str]],
) -> list[dict[str, object]]:
    baseline_index = {key_individual(row): row for row in baseline_rows}
    rows: list[dict[str, object]] = []
    for dreambooth_row in dreambooth_rows:
        key = key_individual(dreambooth_row)
        baseline_row = baseline_index.get(key)
        if baseline_row is None:
            continue
        rows.append(
            {
                "concept": dreambooth_row["concept"],
                "concept_safe": dreambooth_row["concept_safe"],
                "group": dreambooth_row["group"],
                "condition": dreambooth_row["condition"],
                "dreambooth_clip": to_float(dreambooth_row, "clip_mean"),
                "baseline_clip": to_float(baseline_row, "clip_mean"),
                "delta_clip": to_float(dreambooth_row, "clip_mean") - to_float(baseline_row, "clip_mean"),
                "dreambooth_fid": to_float(dreambooth_row, "fid_mean"),
                "baseline_fid": to_float(baseline_row, "fid_mean"),
                "delta_fid": to_float(dreambooth_row, "fid_mean") - to_float(baseline_row, "fid_mean"),
                "dreambooth_kid": to_float(dreambooth_row, "kid_mean"),
                "baseline_kid": to_float(baseline_row, "kid_mean"),
                "delta_kid": to_float(dreambooth_row, "kid_mean") - to_float(baseline_row, "kid_mean"),
                "dreambooth_clip_ci95": format_ci(
                    to_float(dreambooth_row, "clip_mean"),
                    to_float(dreambooth_row, "clip_ci95_low"),
                    to_float(dreambooth_row, "clip_ci95_high"),
                ),
                "baseline_clip_ci95": format_ci(
                    to_float(baseline_row, "clip_mean"),
                    to_float(baseline_row, "clip_ci95_low"),
                    to_float(baseline_row, "clip_ci95_high"),
                ),
                "dreambooth_fid_ci95": format_ci(
                    to_float(dreambooth_row, "fid_mean"),
                    to_float(dreambooth_row, "fid_ci95_low"),
                    to_float(dreambooth_row, "fid_ci95_high"),
                ),
                "baseline_fid_ci95": format_ci(
                    to_float(baseline_row, "fid_mean"),
                    to_float(baseline_row, "fid_ci95_low"),
                    to_float(baseline_row, "fid_ci95_high"),
                ),
                "dreambooth_kid_ci95": format_ci(
                    to_float(dreambooth_row, "kid_mean"),
                    to_float(dreambooth_row, "kid_ci95_low"),
                    to_float(dreambooth_row, "kid_ci95_high"),
                ),
                "baseline_kid_ci95": format_ci(
                    to_float(baseline_row, "kid_mean"),
                    to_float(baseline_row, "kid_ci95_low"),
                    to_float(baseline_row, "kid_ci95_high"),
                ),
            }
        )
    rows.sort(key=lambda row: (str(row["group"]), str(row["concept_safe"]), str(row["condition"])))
    return rows


def build_group_comparison(
    dreambooth_rows: list[dict[str, str]],
    baseline_rows: list[dict[str, str]],
) -> list[dict[str, object]]:
    baseline_index = {key_group(row): row for row in baseline_rows}
    rows: list[dict[str, object]] = []
    for dreambooth_row in dreambooth_rows:
        key = key_group(dreambooth_row)
        baseline_row = baseline_index.get(key)
        if baseline_row is None:
            continue
        rows.append(
            {
                "group": dreambooth_row["group"],
                "condition": dreambooth_row["condition"],
                "num_concepts": dreambooth_row["num_concepts"],
                "dreambooth_clip": to_float(dreambooth_row, "clip_group_mean"),
                "baseline_clip": to_float(baseline_row, "clip_group_mean"),
                "delta_clip": to_float(dreambooth_row, "clip_group_mean") - to_float(baseline_row, "clip_group_mean"),
                "dreambooth_fid": to_float(dreambooth_row, "fid_group_mean"),
                "baseline_fid": to_float(baseline_row, "fid_group_mean"),
                "delta_fid": to_float(dreambooth_row, "fid_group_mean") - to_float(baseline_row, "fid_group_mean"),
                "dreambooth_kid": to_float(dreambooth_row, "kid_group_mean"),
                "baseline_kid": to_float(baseline_row, "kid_group_mean"),
                "delta_kid": to_float(dreambooth_row, "kid_group_mean") - to_float(baseline_row, "kid_group_mean"),
                "dreambooth_clip_ci95": format_ci(
                    to_float(dreambooth_row, "clip_group_mean"),
                    to_float(dreambooth_row, "clip_group_ci95_low"),
                    to_float(dreambooth_row, "clip_group_ci95_high"),
                ),
                "baseline_clip_ci95": format_ci(
                    to_float(baseline_row, "clip_group_mean"),
                    to_float(baseline_row, "clip_group_ci95_low"),
                    to_float(baseline_row, "clip_group_ci95_high"),
                ),
                "dreambooth_fid_ci95": format_ci(
                    to_float(dreambooth_row, "fid_group_mean"),
                    to_float(dreambooth_row, "fid_group_ci95_low"),
                    to_float(dreambooth_row, "fid_group_ci95_high"),
                ),
                "baseline_fid_ci95": format_ci(
                    to_float(baseline_row, "fid_group_mean"),
                    to_float(baseline_row, "fid_group_ci95_low"),
                    to_float(baseline_row, "fid_group_ci95_high"),
                ),
                "dreambooth_kid_ci95": format_ci(
                    to_float(dreambooth_row, "kid_group_mean"),
                    to_float(dreambooth_row, "kid_group_ci95_low"),
                    to_float(dreambooth_row, "kid_group_ci95_high"),
                ),
                "baseline_kid_ci95": format_ci(
                    to_float(baseline_row, "kid_group_mean"),
                    to_float(baseline_row, "kid_group_ci95_low"),
                    to_float(baseline_row, "kid_group_ci95_high"),
                ),
            }
        )
    rows.sort(key=lambda row: (str(row["group"]), str(row["condition"])))
    return rows


def rows_to_markdown(rows: list[dict[str, object]], columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = ["| " + " | ".join(str(row.get(column, "")) for column in columns) + " |" for row in rows]
    return "\n".join([header, separator] + body) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare DreamBooth+LoRA against baseline outputs.")
    parser.add_argument("--dreambooth-root", type=Path, default=Path.cwd() / "experiments_outputs")
    parser.add_argument("--baseline-root", type=Path, default=Path.cwd() / "baseline_outputs")
    parser.add_argument("--output-root", type=Path, default=Path.cwd() / "comparison_outputs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    dreambooth_main = read_csv(args.dreambooth_root / "metrics_dreambooth.csv")
    baseline_main = read_csv(args.baseline_root / "metrics_baseline.csv")
    dreambooth_group = read_csv(args.dreambooth_root / "metrics_groupwise_dreambooth.csv")
    baseline_group = read_csv(args.baseline_root / "metrics_groupwise_baseline.csv")

    individual_rows = build_individual_comparison(dreambooth_main, baseline_main)
    group_rows = build_group_comparison(dreambooth_group, baseline_group)

    individual_fields = [
        "concept",
        "concept_safe",
        "group",
        "condition",
        "dreambooth_clip",
        "baseline_clip",
        "delta_clip",
        "dreambooth_fid",
        "baseline_fid",
        "delta_fid",
        "dreambooth_kid",
        "baseline_kid",
        "delta_kid",
        "dreambooth_clip_ci95",
        "baseline_clip_ci95",
        "dreambooth_fid_ci95",
        "baseline_fid_ci95",
        "dreambooth_kid_ci95",
        "baseline_kid_ci95",
    ]
    group_fields = [
        "group",
        "condition",
        "num_concepts",
        "dreambooth_clip",
        "baseline_clip",
        "delta_clip",
        "dreambooth_fid",
        "baseline_fid",
        "delta_fid",
        "dreambooth_kid",
        "baseline_kid",
        "delta_kid",
        "dreambooth_clip_ci95",
        "baseline_clip_ci95",
        "dreambooth_fid_ci95",
        "baseline_fid_ci95",
        "dreambooth_kid_ci95",
        "baseline_kid_ci95",
    ]

    write_csv(args.output_root / "comparison_individual.csv", individual_fields, individual_rows)
    write_csv(args.output_root / "comparison_groupwise.csv", group_fields, group_rows)

    individual_markdown = rows_to_markdown(
        individual_rows,
        [
            "concept",
            "group",
            "condition",
            "dreambooth_clip_ci95",
            "baseline_clip_ci95",
            "dreambooth_fid_ci95",
            "baseline_fid_ci95",
            "dreambooth_kid_ci95",
            "baseline_kid_ci95",
        ],
    )
    group_markdown = rows_to_markdown(
        group_rows,
        [
            "group",
            "condition",
            "num_concepts",
            "dreambooth_clip_ci95",
            "baseline_clip_ci95",
            "dreambooth_fid_ci95",
            "baseline_fid_ci95",
            "dreambooth_kid_ci95",
            "baseline_kid_ci95",
        ],
    )

    (args.output_root / "comparison_individual.md").write_text(individual_markdown)
    (args.output_root / "comparison_groupwise.md").write_text(group_markdown)


if __name__ == "__main__":
    main()
