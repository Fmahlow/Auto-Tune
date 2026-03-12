#!/usr/bin/env python3

from __future__ import annotations

import csv
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image, ImageDraw
from torchmetrics.functional.multimodal import clip_score
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance
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


@dataclass(frozen=True)
class Concept:
    name: str
    folder: Path
    safe_name: str
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


def compute_fid(real_images: torch.Tensor, generated_images: torch.Tensor) -> float:
    metric = FrechetInceptionDistance(normalize=False)
    metric.update(real_images, real=True)
    metric.update(generated_images, real=False)
    return float(metric.compute().item())


def compute_kid(real_images: torch.Tensor, generated_images: torch.Tensor, subset_size: int) -> float:
    metric = KernelInceptionDistance(subset_size=subset_size, normalize=False)
    metric.update(real_images, real=True)
    metric.update(generated_images, real=False)
    mean_value, _ = metric.compute()
    return float(mean_value.item())


def bootstrap_distribution(
    metric_name: str,
    real_images: torch.Tensor,
    generated_images: torch.Tensor,
    bootstrap_samples: int,
    seed: int,
    kid_subset_size: int,
) -> dict[str, float]:
    if metric_name == "fid":
        point_estimate = compute_fid(real_images, generated_images)
    elif metric_name == "kid":
        point_estimate = compute_kid(real_images, generated_images, kid_subset_size)
    else:
        raise ValueError(f"Unsupported metric: {metric_name}")

    if bootstrap_samples <= 1:
        return {
            "mean": point_estimate,
            "std": 0.0,
            "ci95_low": point_estimate,
            "ci95_high": point_estimate,
        }

    rng = np.random.default_rng(seed)
    real_count = real_images.shape[0]
    generated_count = generated_images.shape[0]
    metric_values = np.empty(bootstrap_samples, dtype=np.float64)

    for idx in range(bootstrap_samples):
        real_indices = torch.tensor(rng.integers(0, real_count, size=real_count), dtype=torch.long)
        generated_indices = torch.tensor(rng.integers(0, generated_count, size=generated_count), dtype=torch.long)
        real_sample = real_images.index_select(0, real_indices)
        generated_sample = generated_images.index_select(0, generated_indices)
        if metric_name == "fid":
            metric_values[idx] = compute_fid(real_sample, generated_sample)
        else:
            metric_values[idx] = compute_kid(real_sample, generated_sample, kid_subset_size)

    return {
        "mean": point_estimate,
        "std": float(metric_values.std(ddof=1)),
        "ci95_low": float(np.percentile(metric_values, 2.5)),
        "ci95_high": float(np.percentile(metric_values, 97.5)),
    }


def compute_clip_statistics(folder: Path, prompt_text: str, bootstrap_samples: int, seed: int) -> dict[str, float]:
    scores: list[float] = []
    for image_path in list_image_files(folder):
        image = Image.open(image_path).convert("RGB")
        image_array = np.asarray(image).astype("float32")
        score = clip_score(
            torch.from_numpy(image_array).permute(2, 0, 1).unsqueeze(0),
            [prompt_text],
            model_name_or_path="openai/clip-vit-base-patch16",
        ).detach()
        scores.append(float(score))
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
