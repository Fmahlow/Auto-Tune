#!/usr/bin/env python3
"""Two-stage pipeline: a single sequential trainer feeds a single generator, so at most
one training job and one generation job ever run concurrently on the GPU -- never two
trainings at once (that combination was measured to be *slower* than fully sequential:
two concurrent DreamBooth trainings took 73 min for 3000 steps vs ~27 min solo).

DreamBooth (train then generate) runs for all remaining concepts first, so every
Textual Inversion job can reuse DreamBooth's without_finetuning images
(--without-finetuning-source) instead of regenerating them.
"""
from __future__ import annotations

import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO = Path("/workspace/Auto-Tune")
DATA_ROOT = "/workspace/data"
DB_ROOT = "/workspace/outputs/dreambooth"
TI_ROOT = "/workspace/outputs/textual_inversion"
LOG_DIR = Path("/workspace/logs/jobs")
KID_SUBSET_SIZE = 20

ENV = dict(os.environ)
ENV.setdefault("DIFFUSERS_REPO", "/workspace/diffusers")
ENV.setdefault("TOKENIZERS_PARALLELISM", "false")

# chamanto, chaneques, cuscuz: DreamBooth train+generate already completed by the
# previous (buggy) run's orphaned self-contained jobs left running to finish naturally.
DB_REMAINING = ["jian", "lokum", "paçoca", "patuá", "saci"]
TI_REMAINING = ["chamanto", "chaneques", "cuscuz", "jian", "lokum", "paçoca", "patuá", "saci"]


def db_train_cmd(concept: str) -> list[str]:
    return [
        "python3", "experiments.py",
        "--data-root", DATA_ROOT, "--output-root", DB_ROOT,
        "--concepts", concept,
        "--enable-xformers", "--use-8bit-adam",
        "--train-batch-size", "2", "--gradient-accumulation-steps", "1",
        "--num-images", "1000", "--checkpoint-num-images", "100", "--gen-batch-size", "4",
        "--kid-subset-size", str(KID_SUBSET_SIZE),
        "--skip-generate", "--skip-eval",
    ]


def db_gen_cmd(concept: str) -> list[str]:
    return [
        "python3", "experiments.py",
        "--data-root", DATA_ROOT, "--output-root", DB_ROOT,
        "--concepts", concept,
        "--enable-xformers", "--use-8bit-adam",
        "--num-images", "1000", "--checkpoint-num-images", "100", "--gen-batch-size", "4",
        "--kid-subset-size", str(KID_SUBSET_SIZE),
        "--skip-train", "--skip-eval",
    ]


def ti_train_cmd(concept: str) -> list[str]:
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
        "--skip-generate", "--skip-eval",
    ]


def ti_gen_cmd(concept: str) -> list[str]:
    return [
        "python3", "baseline.py",
        "--data-root", DATA_ROOT, "--output-root", TI_ROOT,
        "--concepts", concept,
        "--enable-xformers", "--mixed-precision", "bf16",
        "--num-images", "1000", "--gen-batch-size", "4",
        "--kid-subset-size", str(KID_SUBSET_SIZE),
        "--checkpoints",
        "--without-finetuning-source", DB_ROOT,
        "--skip-train", "--skip-eval",
    ]


def run(name: str, cmd: list[str]) -> int:
    log_path = LOG_DIR / f"{name}.log"
    t0 = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] START {name}: {' '.join(cmd)}", flush=True)
    with log_path.open("w") as handle:
        proc = subprocess.run(cmd, cwd=str(REPO), stdout=handle, stderr=subprocess.STDOUT, env=ENV)
    dt = time.time() - t0
    status = "OK" if proc.returncode == 0 else f"FAIL(rc={proc.returncode})"
    print(f"[{time.strftime('%H:%M:%S')}] END {name} {status} in {dt / 60:.1f} min -> {log_path}", flush=True)
    return proc.returncode


def main() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    gen_queue: "queue.Queue[tuple[str, list[str]] | None]" = queue.Queue()
    train_results: dict[str, int] = {}
    gen_results: dict[str, int] = {}

    def generator_worker() -> None:
        while True:
            item = gen_queue.get()
            if item is None:
                break
            name, cmd = item
            gen_results[name] = run(name, cmd)

    gen_thread = threading.Thread(target=generator_worker, daemon=True)
    gen_thread.start()

    train_jobs: list[tuple[str, list[str], str, list[str]]] = []
    for concept in DB_REMAINING:
        train_jobs.append((f"db_train_{concept}", db_train_cmd(concept), f"db_gen_{concept}", db_gen_cmd(concept)))
    for concept in TI_REMAINING:
        train_jobs.append((f"ti_train_{concept}", ti_train_cmd(concept), f"ti_gen_{concept}", ti_gen_cmd(concept)))

    print(f"[{time.strftime('%H:%M:%S')}] Total concept-jobs: {len(train_jobs)} "
          f"(DB remaining: {DB_REMAINING}, TI remaining: {TI_REMAINING})", flush=True)
    t_start = time.time()
    for train_name, train_cmd, gen_name, gen_cmd in train_jobs:
        rc = run(train_name, train_cmd)
        train_results[train_name] = rc
        if rc == 0:
            gen_queue.put((gen_name, gen_cmd))
        else:
            print(f"[{time.strftime('%H:%M:%S')}] SKIPPING generation for {gen_name} "
                  f"because training failed", flush=True)

    gen_queue.put(None)
    gen_thread.join()

    total_min = (time.time() - t_start) / 60
    failed = [n for n, rc in {**train_results, **gen_results}.items() if rc != 0]
    print(f"[{time.strftime('%H:%M:%S')}] ALL GENERATION/TRAINING JOBS DONE in {total_min:.1f} min. "
          f"Failed: {failed if failed else 'none'}", flush=True)
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
