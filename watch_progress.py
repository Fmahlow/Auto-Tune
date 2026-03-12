#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def format_seconds(value: float | None) -> str:
    if value is None:
        return "unknown"
    seconds = int(value)
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def render(progress_path: Path) -> str:
    data = json.loads(progress_path.read_text())
    return "\n".join(
        [
            f"run:      {data.get('run_name', '')}",
            f"stage:    {data.get('current_stage', '')}",
            f"concept:  {data.get('current_concept', '')}",
            f"detail:   {data.get('current_detail', '')}",
            f"progress: {data.get('completed_steps', 0)}/{data.get('total_steps', 0)} ({data.get('progress_percent', 0)}%)",
            f"elapsed:  {format_seconds(data.get('elapsed_seconds'))}",
            f"eta:      {format_seconds(data.get('eta_seconds'))}",
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Watch progress.json for long experiment runs.")
    parser.add_argument("progress_file", type=Path, help="Path to progress.json")
    parser.add_argument("--interval", type=float, default=5.0, help="Refresh interval in seconds.")
    parser.add_argument("--once", action="store_true", help="Print once and exit.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    while True:
        if args.progress_file.exists():
            print("\033c", end="")
            print(render(args.progress_file))
        else:
            print("\033c", end="")
            print(f"Waiting for {args.progress_file} ...")
        if args.once:
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
