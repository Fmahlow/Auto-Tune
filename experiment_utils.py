#!/usr/bin/env python3

from __future__ import annotations

import csv
import json
import math
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image, ImageDraw
from torchmetrics.multimodal import CLIPScore
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision.transforms import functional as F


VISUAL_DESCRIPTIONS = {
    "patuá": "A small amulet or charm, often worn as a necklace, made of fabric or leather.",
    "patua": "A small amulet or charm, often worn as a necklace, made of fabric or leather.",
    "cuscuz": "A plate of golden, grainy steamed cornmeal, often served with cheese or meat.",
    "chaneques": "Mythical small humanoid creatures, resembling goblins, from Mexican folklore.",
    "chamanto": "A traditional Chilean poncho with intricate patterns, worn over the shoulders.",
    "lokum": "Turkish delight, a gelatinous sweet dusted with powdered sugar, in various colors.",
    "paçoca": (
        "A cylindrical or rectangular Brazilian sweet made of ground peanuts, sugar, and salt, "
        "with a crumbly and slightly rough texture, typically light brown in color."
    ),
    "pacoca": (
        "A cylindrical or rectangular Brazilian sweet made of ground peanuts, sugar, and salt, "
        "with a crumbly and slightly rough texture, typically light brown in color."
    ),
    "jian": "A straight double-edged Chinese sword with a narrow, elegant blade.",
    "saci": "A one-legged trickster from Brazilian folklore, with dark skin, wearing a red cap and smoking a pipe.",
}

CONCEPT_GROUPS = {
    "cuscuz": "food",
    "lokum": "food",
    "pacoca": "food",
    "patua": "artifact_object_clothing",
    "jian": "artifact_object_clothing",
    "chamanto": "artifact_object_clothing",
    "saci": "folklore_being",
    "chaneques": "folklore_being",
}


@dataclass(frozen=True)
class Concept:
    name: str
    folder: Path
    safe_name: str
    group: str
    prompt_base: str
    visual_description: str


def normalize_name(raw: str) -> str:
    cleaned = raw.strip().lower()
    cleaned = cleaned.replace("ç", "c").replace("á", "a").replace("ã", "a").replace("â", "a")
    cleaned = cleaned.replace("é", "e").replace("ê", "e").replace("í", "i").replace("ó", "o")
    cleaned = cleaned.replace("ô", "o").replace("õ", "o").replace("ú", "u")
    cleaned = re.sub(r"\s+", "_", cleaned)
    cleaned = re.sub(r"[^a-z0-9_]+", "", cleaned)
    return cleaned


def is_image_file(path: Path) -> bool:
    return path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def discover_concepts(data_root: Path, requested: list[str] | None) -> list[Concept]:
    folders = [p for p in data_root.iterdir() if p.is_dir() and not p.name.startswith(".")]
    requested_names = {normalize_name(name) for name in requested} if requested else None
    concepts: list[Concept] = []
    for folder in sorted(folders, key=lambda item: item.name.lower()):
        if not any(is_image_file(path) for path in folder.iterdir() if path.is_file()):
            continue
        safe_name = normalize_name(folder.name)
        if requested_names and safe_name not in requested_names:
            continue
        prompt_base = f"a photo of {folder.name}"
        visual_description = VISUAL_DESCRIPTIONS.get(folder.name.lower(), VISUAL_DESCRIPTIONS.get(safe_name, prompt_base))
        concepts.append(
            Concept(
                name=folder.name,
                folder=folder,
                safe_name=safe_name,
                group=CONCEPT_GROUPS.get(safe_name, "other"),
                prompt_base=prompt_base,
                visual_description=visual_description,
            )
        )
    return concepts


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


class ProgressTracker:
    def __init__(self, output_root: Path, run_name: str, total_steps: int) -> None:
        self.output_root = output_root
        self.run_name = run_name
        self.total_steps = max(total_steps, 1)
        self.completed_steps = 0
        self.current_stage = "initializing"
        self.current_concept = ""
        self.current_detail = ""
        self.started_at = time.time()
        self.progress_path = output_root / "progress.json"
        self.log_path = output_root / "progress.log"
        self.output_root.mkdir(parents=True, exist_ok=True)
        self._flush()

    def set_stage(self, stage: str, concept: str = "", detail: str = "") -> None:
        self.current_stage = stage
        self.current_concept = concept
        self.current_detail = detail
        self._flush()

    def advance(self, amount: int = 1, stage: str | None = None, concept: str | None = None, detail: str | None = None) -> None:
        self.completed_steps = min(self.total_steps, self.completed_steps + amount)
        if stage is not None:
            self.current_stage = stage
        if concept is not None:
            self.current_concept = concept
        if detail is not None:
            self.current_detail = detail
        self._flush()

    def log(self, message: str) -> None:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        with self.log_path.open("a") as handle:
            handle.write(f"[{timestamp}] {message}\n")

    def complete(self) -> None:
        self.completed_steps = self.total_steps
        self.current_stage = "completed"
        self.current_detail = "finished"
        self._flush()

    def _flush(self) -> None:
        elapsed = max(time.time() - self.started_at, 0.0)
        fraction = self.completed_steps / self.total_steps
        eta_seconds = None
        if self.completed_steps > 0 and self.completed_steps < self.total_steps:
            eta_seconds = elapsed * (self.total_steps - self.completed_steps) / self.completed_steps
        payload = {
            "run_name": self.run_name,
            "current_stage": self.current_stage,
            "current_concept": self.current_concept,
            "current_detail": self.current_detail,
            "completed_steps": self.completed_steps,
            "total_steps": self.total_steps,
            "progress_fraction": fraction,
            "progress_percent": round(fraction * 100.0, 2),
            "elapsed_seconds": round(elapsed, 2),
            "eta_seconds": None if eta_seconds is None else round(eta_seconds, 2),
        }
        self.progress_path.write_text(json.dumps(payload, indent=2) + "\n")


def list_image_files(folder: Path) -> list[Path]:
    return [path for path in sorted(folder.iterdir()) if path.is_file() and is_image_file(path)]


def bootstrap_summary(values: Iterable[float], bootstrap_samples: int, seed: int) -> dict[str, float]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        return {"mean": math.nan, "std": math.nan, "ci95_low": math.nan, "ci95_high": math.nan}
    if array.size == 1 or bootstrap_samples <= 1:
        value = float(array.mean())
        return {"mean": value, "std": 0.0, "ci95_low": value, "ci95_high": value}

    rng = np.random.default_rng(seed)
    means = np.empty(bootstrap_samples, dtype=np.float64)
    for idx in range(bootstrap_samples):
        sample = rng.choice(array, size=array.size, replace=True)
        means[idx] = sample.mean()
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)),
        "ci95_low": float(np.percentile(means, 2.5)),
        "ci95_high": float(np.percentile(means, 97.5)),
    }


def load_image_tensor_uint8(file_path: Path, size: int) -> torch.Tensor:
    image = Image.open(file_path).convert("RGB")
    image = F.resize(image, [size, size])
    array = np.asarray(image, dtype=np.uint8)
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)


def load_folder_tensor_uint8(folder: Path, size: int) -> torch.Tensor:
    tensors = [load_image_tensor_uint8(path, size) for path in list_image_files(folder)]
    if not tensors:
        raise ValueError(f"No images found in {folder}")
    return torch.cat(tensors, dim=0)


def _extract_inception_features(images_uint8: torch.Tensor) -> torch.Tensor:
    """Extract InceptionV3 2048-d features from uint8 NCHW images. Uses GPU when available.
    Returns a double-precision tensor kept on the extraction device (GPU when available) so
    the whole bootstrap loop below can run without any GPU<->CPU round trips."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    extractor = FrechetInceptionDistance(feature=2048, normalize=False).to(device)
    extractor.eval()
    batch_size = 32
    all_features: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, len(images_uint8), batch_size):
            batch = images_uint8[start : start + batch_size].to(device)
            feats = extractor.inception(batch)
            all_features.append(feats.double())
    return torch.cat(all_features, dim=0)


def _trace_sqrtm_product_psd(sigma_r: torch.Tensor, sigma_g: torch.Tensor) -> torch.Tensor:
    """trace(sqrtm(sigma_r @ sigma_g)) for symmetric PSD sigma_r, sigma_g, computed via
    eigendecomposition instead of scipy.linalg.sqrtm's CPU Schur algorithm.
    scipy.linalg.sqrtm on a 2048x2048 matrix can take seconds per call; with a 1000-sample
    bootstrap over dozens of folders that becomes hours. The identity
    trace(sqrtm(A @ B)) == trace(sqrtm(A^{1/2} @ B @ A^{1/2})) for symmetric PSD A, B lets us
    replace the general matrix square root with two symmetric eigh calls (fast, GPU-able) and
    a sum of sqrt(eigenvalues) -- the full sqrtm matrix itself is never needed, only its trace.
    """
    sr = (sigma_r + sigma_r.T) / 2
    sg = (sigma_g + sigma_g.T) / 2
    eigvals_r, eigvecs_r = torch.linalg.eigh(sr)
    eigvals_r = torch.clamp(eigvals_r, min=0)
    sr_sqrt = (eigvecs_r * torch.sqrt(eigvals_r)) @ eigvecs_r.T
    b = sr_sqrt @ sg @ sr_sqrt
    b = (b + b.T) / 2
    eigvals_b = torch.linalg.eigvalsh(b)
    eigvals_b = torch.clamp(eigvals_b, min=0)
    return torch.sum(torch.sqrt(eigvals_b))


def _cov(feat: torch.Tensor) -> torch.Tensor:
    if feat.shape[0] <= 1:
        d = feat.shape[1]
        return torch.zeros((d, d), device=feat.device, dtype=feat.dtype)
    centered = feat - feat.mean(dim=0, keepdim=True)
    return (centered.T @ centered) / (feat.shape[0] - 1)


def _fid_from_features(real_feat: torch.Tensor, gen_feat: torch.Tensor) -> float:
    mu_r, mu_g = real_feat.mean(0), gen_feat.mean(0)
    sigma_r = _cov(real_feat)
    sigma_g = _cov(gen_feat)
    diff = mu_r - mu_g
    trace_sqrt = _trace_sqrtm_product_psd(sigma_r, sigma_g)
    result = torch.dot(diff, diff) + torch.trace(sigma_r) + torch.trace(sigma_g) - 2.0 * trace_sqrt
    return float(result.item())


def _kid_from_features(real_feat: torch.Tensor, gen_feat: torch.Tensor) -> float:
    """Unbiased MMD with cubic polynomial kernel k(x,y) = (x·y/d + 1)^3."""
    d = real_feat.shape[1]
    m, n = real_feat.shape[0], gen_feat.shape[0]
    kxx = ((real_feat @ real_feat.T) / d + 1.0) ** 3
    kyy = ((gen_feat @ gen_feat.T) / d + 1.0) ** 3
    kxy = ((real_feat @ gen_feat.T) / d + 1.0) ** 3
    kxx.fill_diagonal_(0.0)
    kyy.fill_diagonal_(0.0)
    result = kxx.sum() / (m * (m - 1)) + kyy.sum() / (n * (n - 1)) - 2.0 * kxy.mean()
    return float(result.item())


def bootstrap_distribution(
    metric_name: str,
    real_images: torch.Tensor,
    generated_images: torch.Tensor,
    bootstrap_samples: int,
    seed: int,
    kid_subset_size: int,
) -> dict[str, float]:
    """Extract InceptionV3 features once on GPU, then bootstrap over features -- resampling,
    covariance, and the FID trace term all stay on the GPU as torch tensors so a 1000-sample
    bootstrap takes seconds instead of hours."""
    real_feat = _extract_inception_features(real_images)
    gen_feat = _extract_inception_features(generated_images)
    device = real_feat.device

    compute_metric = _fid_from_features if metric_name == "fid" else _kid_from_features
    point_estimate = compute_metric(real_feat, gen_feat)

    if bootstrap_samples <= 1:
        return {"mean": point_estimate, "std": 0.0, "ci95_low": point_estimate, "ci95_high": point_estimate}

    rng = np.random.default_rng(seed)
    metric_values = np.empty(bootstrap_samples, dtype=np.float64)
    for idx in range(bootstrap_samples):
        r_idx = torch.from_numpy(rng.integers(0, len(real_feat), size=len(real_feat))).to(device)
        g_idx = torch.from_numpy(rng.integers(0, len(gen_feat), size=len(gen_feat))).to(device)
        metric_values[idx] = compute_metric(real_feat[r_idx], gen_feat[g_idx])

    return {
        "mean": point_estimate,
        "std": float(metric_values.std(ddof=1)),
        "ci95_low": float(np.percentile(metric_values, 2.5)),
        "ci95_high": float(np.percentile(metric_values, 97.5)),
    }


def compute_clip_statistics(folder: Path, prompt_text: str, bootstrap_samples: int, seed: int) -> dict[str, float]:
    """Compute CLIP scores loading the model once on GPU, processing one image at a time."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_metric = CLIPScore(model_name_or_path="openai/clip-vit-base-patch16").to(device)
    clip_metric.eval()
    scores: list[float] = []
    with torch.no_grad():
        for image_path in list_image_files(folder):
            image = Image.open(image_path).convert("RGB")
            arr = np.asarray(image, dtype=np.uint8)
            tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
            score = clip_metric(tensor, [prompt_text])
            scores.append(float(score.detach().cpu()))
    summary = bootstrap_summary(scores, bootstrap_samples, seed)
    summary["sample_count"] = len(scores)
    return summary


def evaluate_generated_folder(
    concept: Concept,
    generated_folder: Path,
    condition_label: str,
    method_label: str,
    bootstrap_samples: int,
    metric_seed: int,
    fid_resize: int,
    kid_subset_size: int,
) -> dict[str, object]:
    clip_stats = compute_clip_statistics(generated_folder, concept.visual_description, bootstrap_samples, metric_seed)
    real_images = load_folder_tensor_uint8(concept.folder, fid_resize)
    generated_images = load_folder_tensor_uint8(generated_folder, fid_resize)
    fid_stats = bootstrap_distribution("fid", real_images, generated_images, bootstrap_samples, metric_seed + 1, kid_subset_size)
    kid_stats = bootstrap_distribution("kid", real_images, generated_images, bootstrap_samples, metric_seed + 2, kid_subset_size)
    return {
        "concept": concept.name,
        "concept_safe": concept.safe_name,
        "group": concept.group,
        "method": method_label,
        "condition": condition_label,
        "folder": str(generated_folder),
        "sample_count": clip_stats["sample_count"],
        "clip_mean": clip_stats["mean"],
        "clip_std": clip_stats["std"],
        "clip_ci95_low": clip_stats["ci95_low"],
        "clip_ci95_high": clip_stats["ci95_high"],
        "fid_mean": fid_stats["mean"],
        "fid_std": fid_stats["std"],
        "fid_ci95_low": fid_stats["ci95_low"],
        "fid_ci95_high": fid_stats["ci95_high"],
        "kid_mean": kid_stats["mean"],
        "kid_std": kid_stats["std"],
        "kid_ci95_low": kid_stats["ci95_low"],
        "kid_ci95_high": kid_stats["ci95_high"],
    }


def aggregate_metrics_by_group(rows: list[dict[str, object]], bootstrap_samples: int, seed: int) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, str], list[dict[str, object]]] = {}
    for row in rows:
        key = (str(row["group"]), str(row["method"]), str(row["condition"]))
        grouped.setdefault(key, []).append(row)

    aggregated_rows: list[dict[str, object]] = []
    metric_names = ["clip", "fid", "kid"]
    for (group, method, condition), group_rows in sorted(grouped.items()):
        aggregated = {
            "group": group,
            "method": method,
            "condition": condition,
            "num_concepts": len(group_rows),
            "total_sample_count": int(sum(int(row["sample_count"]) for row in group_rows)),
        }
        for metric_index, metric_name in enumerate(metric_names):
            values = [float(row[f"{metric_name}_mean"]) for row in group_rows]
            arr = np.array(values)
            if len(values) < 4:
                # Too few concepts for bootstrap CI; report min/max instead.
                aggregated[f"{metric_name}_group_mean"] = float(arr.mean())
                aggregated[f"{metric_name}_group_std"] = float(arr.std(ddof=1)) if len(values) > 1 else 0.0
                aggregated[f"{metric_name}_group_ci95_low"] = float(arr.min())
                aggregated[f"{metric_name}_group_ci95_high"] = float(arr.max())
            else:
                stats = bootstrap_summary(values, bootstrap_samples, seed + metric_index)
                aggregated[f"{metric_name}_group_mean"] = stats["mean"]
                aggregated[f"{metric_name}_group_std"] = stats["std"]
                aggregated[f"{metric_name}_group_ci95_low"] = stats["ci95_low"]
                aggregated[f"{metric_name}_group_ci95_high"] = stats["ci95_high"]
        aggregated_rows.append(aggregated)
    return aggregated_rows


def build_contact_sheet(rows: list[tuple[str, list[Path]]], destination: Path, tile_size: int = 256) -> None:
    if not rows:
        return
    label_height = 28
    row_gap = 16
    col_gap = 16
    max_cols = max(len(paths) for _, paths in rows)
    width = 180 + max_cols * tile_size + max(0, max_cols - 1) * col_gap
    height = len(rows) * (tile_size + label_height) + max(0, len(rows) - 1) * row_gap
    canvas = Image.new("RGB", (width, height), color=(245, 245, 245))
    draw = ImageDraw.Draw(canvas)

    for row_index, (label, paths) in enumerate(rows):
        y = row_index * (tile_size + label_height + row_gap)
        draw.text((8, y + tile_size // 2), label, fill=(20, 20, 20))
        for col_index, path in enumerate(paths):
            x = 180 + col_index * (tile_size + col_gap)
            image = Image.open(path).convert("RGB")
            image = image.resize((tile_size, tile_size))
            canvas.paste(image, (x, y))
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination)


def create_qualitative_sheet(
    concept: Concept,
    reference_folder: Path,
    condition_folders: list[tuple[str, Path]],
    destination: Path,
    samples_per_row: int,
    seed: int,
) -> None:
    rng = random.Random(seed)
    rows: list[tuple[str, list[Path]]] = []
    reference_files = list_image_files(reference_folder)
    if reference_files:
        sample_count = min(samples_per_row, len(reference_files))
        rows.append(("reference", rng.sample(reference_files, sample_count)))
    for label, folder in condition_folders:
        generated_files = list_image_files(folder)
        if not generated_files:
            continue
        sample_count = min(samples_per_row, len(generated_files))
        rows.append((label, rng.sample(generated_files, sample_count)))
    build_contact_sheet(rows, destination)


def create_human_eval_package(
    concepts: list[Concept],
    condition_map: dict[str, list[tuple[str, Path]]],
    output_root: Path,
    samples_per_condition: int,
    seed: int,
) -> tuple[Path, Path, Path]:
    rng = random.Random(seed)
    manifest_rows: list[dict[str, object]] = []
    key_rows: list[dict[str, object]] = []
    blinded_labels = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

    for concept in concepts:
        conditions = condition_map.get(concept.safe_name, [])
        if not conditions:
            continue
        shuffled = conditions[:]
        rng.shuffle(shuffled)
        for idx, (condition_name, folder) in enumerate(shuffled):
            blind_id = blinded_labels[idx]
            key_rows.append(
                {
                    "concept": concept.name,
                    "concept_safe": concept.safe_name,
                    "blind_id": blind_id,
                    "condition": condition_name,
                    "folder": str(folder),
                }
            )
            files = list_image_files(folder)
            if not files:
                continue
            sample_count = min(samples_per_condition, len(files))
            for sample_index, file_path in enumerate(rng.sample(files, sample_count), start=1):
                manifest_rows.append(
                    {
                        "concept": concept.name,
                        "concept_safe": concept.safe_name,
                        "blind_id": blind_id,
                        "sample_id": f"{concept.safe_name}_{blind_id}_{sample_index:03d}",
                        "image_path": str(file_path),
                        "cultural_recognizability": "",
                        "cultural_fidelity": "",
                        "overall_quality": "",
                        "notes": "",
                    }
                )

    human_eval_dir = output_root / "human_eval"
    manifest_path = human_eval_dir / "human_eval_manifest.csv"
    key_path = human_eval_dir / "human_eval_key.csv"
    instructions_path = human_eval_dir / "instructions.txt"

    write_csv(
        manifest_path,
        [
            "concept",
            "concept_safe",
            "blind_id",
            "sample_id",
            "image_path",
            "cultural_recognizability",
            "cultural_fidelity",
            "overall_quality",
            "notes",
        ],
        manifest_rows,
    )
    write_csv(key_path, ["concept", "concept_safe", "blind_id", "condition", "folder"], key_rows)
    instructions_path.parent.mkdir(parents=True, exist_ok=True)
    instructions_path.write_text(
        "\n".join(
            [
                "Human evaluation protocol",
                "Rate each image on a 1-5 scale for cultural recognizability, cultural fidelity, and overall quality.",
                "Use raters familiar with the corresponding cultural concept whenever possible.",
                "Keep blind_id hidden from raters; it maps to the actual condition in human_eval_key.csv.",
                "Record optional free-text notes for obvious artifacts, mismatches, or culturally salient details.",
            ]
        )
        + "\n"
    )
    return manifest_path, key_path, instructions_path
