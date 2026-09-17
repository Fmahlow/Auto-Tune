#!/usr/bin/env python3
"""Orchestrates the full DreamBooth+LoRA and Textual Inversion runs for all concepts
with bounded GPU concurrency. Each job trains (if needed) then generates for a single
concept in one subprocess call to experiments.py / baseline.py, so train->generate is
naturally pipelined per concept. DreamBooth jobs run first (method-major order) so that
every Textual Inversion job can reuse DreamBooth's without_finetuning images instead of
regenerating them (--without-finetuning-source).
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path("/workspace/Auto-Tune")
DATA_ROOT = "/workspace/data"
DB_ROOT = "/workspace/outputs/dreambooth"
TI_ROOT = "/workspace/outputs/textual_inversion"
LOG_DIR = Path("/workspace/logs/jobs")
CONCEPTS = ["chamanto", "chaneques", "cuscuz", "jian", "lokum", "paçoca", "patuá", "saci"]
KID_SUBSET_SIZE = 20
CONCURRENCY = 2

ENV = dict(os.environ)
ENV.setdefault("DIFFUSERS_REPO", "/workspace/diffusers")
ENV.setdefault("TOKENIZERS_PARALLELISM", "false")


def db_cmd(concept: str) -> list[str]:
    return [
        "python3", "experiments.py",
        "--data-root", DATA_ROOT, "--output-root", DB_ROOT,
        "--concepts", concept,
        "--enable-xformers", "--use-8bit-adam",
        "--train-batch-size", "2", "--gradient-accumulation-steps", "1",
        "--num-images", "1000", "--checkpoint-num-images", "100", "--gen-batch-size", "4",
        "--kid-subset-size", str(KID_SUBSET_SIZE),
        "--skip-eval",
    ]


def ti_cmd(concept: str) -> list[str]:
    return [
        "python3", "baseline.py",
        "--data-root", DATA_ROOT, "--output-root", TI_ROOT,
        "--concepts", concept,
        "--enable-xformers", "--mixed-precision", "bf16",
        "--train-batch-size", "2", "--gradient-accumulation-steps", "1",
        "--num-images", "1000", "--gen-batch-size", "4",
        "--kid-subset-size", str(KID_SUBSET_SIZE),
        "--checkpoints",
        "--without-finetuning-source", DB_ROOT,
        "--skip-eval",
    ]


def run_job(name: str, cmd: list[str]) -> tuple[str, int, float]:
    log_path = LOG_DIR / f"{name}.log"
    t0 = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] START {name}: {' '.join(cmd)}", flush=True)
    with log_path.open("w") as handle:
        proc = subprocess.run(cmd, cwd=str(REPO), stdout=handle, stderr=subprocess.STDOUT, env=ENV)
    dt = time.time() - t0
    status = "OK" if proc.returncode == 0 else f"FAIL(rc={proc.returncode})"
    print(f"[{time.strftime('%H:%M:%S')}] END {name} {status} in {dt / 60:.1f} min -> {log_path}", flush=True)
    return name, proc.returncode, dt


def main() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    jobs: list[tuple[str, list[str]]] = []
    for concept in CONCEPTS:
        jobs.append((f"db_{concept}", db_cmd(concept)))
    for concept in CONCEPTS:
        jobs.append((f"ti_{concept}", ti_cmd(concept)))

    print(f"[{time.strftime('%H:%M:%S')}] Total jobs: {len(jobs)}, concurrency={CONCURRENCY}", flush=True)
    t_start = time.time()
    results: dict[str, int] = {}
    with ThreadPoolExecutor(max_workers=CONCURRENCY) as executor:
        futures = [executor.submit(run_job, name, cmd) for name, cmd in jobs]
        for future in futures:
            name, rc, _dt = future.result()
            results[name] = rc

    failed = [name for name, rc in results.items() if rc != 0]
    total_min = (time.time() - t_start) / 60
    print(f"[{time.strftime('%H:%M:%S')}] ALL GENERATION/TRAINING JOBS DONE in {total_min:.1f} min. "
          f"Failed: {failed if failed else 'none'}", flush=True)
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
