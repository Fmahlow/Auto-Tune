#!/usr/bin/env python3
"""
Baseline experiments with Textual Inversion (SDXL), mirroring experiments.ipynb flow.

Pipeline:
1) Train one textual inversion model per concept folder.
2) Generate images without fine-tuning (base model prompt).
3) Generate images with fine-tuning (placeholder token prompt).
4) Optionally generate images for checkpoint embeddings (500..3000 by default).
5) Compute CLIP score CSVs and FID CSVs.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image
from torchmetrics.functional.multimodal import clip_score
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision.transforms import functional as F

from diffusers import DiffusionPipeline, LCMScheduler, UNet2DConditionModel


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
    token: str
    prompt_base: str
    prompt_ti: str
    visual_description: str
    safe_name: str


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
    concepts: list[Concept] = []
    req = {normalize_name(x): x for x in requested} if requested else None

    for folder in sorted(folders, key=lambda x: x.name.lower()):
        images = [f for f in folder.iterdir() if f.is_file() and is_image_file(f)]
        if not images:
            continue

        canonical = normalize_name(folder.name)
        if req and canonical not in req:
            continue

        token = f"<ti_{canonical}>"
        prompt_base = f"a photo of {folder.name}"
        prompt_ti = f"a photo of {token}"
        visual_description = VISUAL_DESCRIPTIONS.get(folder.name.lower(), VISUAL_DESCRIPTIONS.get(canonical, prompt_base))
        concepts.append(
            Concept(
                name=folder.name,
                folder=folder,
                token=token,
                prompt_base=prompt_base,
                prompt_ti=prompt_ti,
                visual_description=visual_description,
                safe_name=canonical,
            )
        )
    return concepts


def run_cmd(cmd: list[str]) -> None:
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def train_textual_inversion(
    concept: Concept,
    output_root: Path,
    train_script: Path,
    model_name: str,
    vae_path: str,
    resolution: int,
    train_batch_size: int,
    grad_accum: int,
    lr: float,
    max_train_steps: int,
    checkpoint_steps: int,
    save_steps: int,
    mixed_precision: str,
    initializer_token: str,
    extra_args: list[str],
) -> Path:
    concept_output = output_root / "training_runs" / concept.safe_name
    concept_output.mkdir(parents=True, exist_ok=True)

    cmd = [
        "accelerate",
        "launch",
        str(train_script),
        f"--pretrained_model_name_or_path={model_name}",
        f"--pretrained_vae_model_name_or_path={vae_path}",
        f"--train_data_dir={concept.folder}",
        f"--output_dir={concept_output}",
        f"--placeholder_token={concept.token}",
        f"--initializer_token={initializer_token}",
        f"--resolution={resolution}",
        f"--train_batch_size={train_batch_size}",
        f"--gradient_accumulation_steps={grad_accum}",
        f"--learning_rate={lr}",
        "--lr_scheduler=constant",
        "--lr_warmup_steps=0",
        f"--max_train_steps={max_train_steps}",
        f"--checkpointing_steps={checkpoint_steps}",
        f"--save_steps={save_steps}",
        "--validation_prompt=" + concept.prompt_ti,
        "--validation_epochs=25",
        "--enable_xformers_memory_efficient_attention",
    ]

    if mixed_precision:
        cmd.append(f"--mixed_precision={mixed_precision}")

    cmd.extend(extra_args)
    run_cmd(cmd)
    return concept_output


def load_pipeline(base_model: str, use_lcm: bool) -> DiffusionPipeline:
    if use_lcm:
        unet = UNet2DConditionModel.from_pretrained(
            "latent-consistency/lcm-sdxl",
            torch_dtype=torch.float16,
            variant="fp16",
        )
        pipe = DiffusionPipeline.from_pretrained(base_model, unet=unet, torch_dtype=torch.float16).to("cuda")
        pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    else:
        pipe = DiffusionPipeline.from_pretrained(base_model, torch_dtype=torch.float16).to("cuda")
    return pipe


def unload_ti_if_supported(pipe: DiffusionPipeline, token: str) -> None:
    if hasattr(pipe, "unload_textual_inversion"):
        try:
            pipe.unload_textual_inversion(token)
        except Exception:
            pass


def generate_images(
    pipe: DiffusionPipeline,
    prompt: str,
    out_dir: Path,
    n_images: int,
    num_inference_steps: int,
    guidance_scale: float,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for i in range(1, n_images + 1):
        image = pipe(
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        ).images[0]
        image.save(out_dir / f"image_{i:03d}.png")


def find_final_embedding(concept_train_dir: Path) -> Path:
    candidates = []
    for pattern in ("learned_embeds.safetensors", "learned_embeds.bin", "learned_embeds.pt"):
        p = concept_train_dir / pattern
        if p.exists():
            candidates.append(p)
    if not candidates:
        raise FileNotFoundError(f"Final embedding file not found in {concept_train_dir}")
    return candidates[0]


def find_checkpoint_embedding(concept_train_dir: Path, step: int) -> Path | None:
    patterns = [
        f"learned_embeds-steps-{step}.safetensors",
        f"learned_embeds-steps-{step}.bin",
        f"learned_embeds-steps-{step}.pt",
        f"checkpoint-{step}/learned_embeds.safetensors",
        f"checkpoint-{step}/learned_embeds.bin",
        f"checkpoint-{step}/learned_embeds.pt",
    ]
    for pattern in patterns:
        p = concept_train_dir / pattern
        if p.exists():
            return p
    return None


def load_images_for_clip(folder: Path, prompt_text: str) -> tuple[list[np.ndarray], list[str]]:
    images: list[np.ndarray] = []
    prompts: list[str] = []
    for image_file in sorted(folder.iterdir()):
        if not image_file.is_file() or not is_image_file(image_file):
            continue
        image = Image.open(image_file).convert("RGB")
        images.append(np.array(image))
        prompts.append(prompt_text)
    return images, prompts


def calculate_clip_scores(images: Iterable[np.ndarray], prompts: Iterable[str]) -> list[float]:
    results: list[float] = []
    for image, prompt in zip(images, prompts):
        image_float = image.astype("float32")
        score = clip_score(
            torch.from_numpy(image_float).permute(2, 0, 1).unsqueeze(0),
            [prompt],
            model_name_or_path="openai/clip-vit-base-patch16",
        ).detach()
        results.append(float(score))
    return results


def load_and_process_for_fid(file_path: Path, size: int) -> torch.Tensor | None:
    try:
        image = Image.open(file_path).convert("RGB")
        image = F.resize(image, [size, size])
        arr = np.array(image)
        return torch.tensor(arr).unsqueeze(0).permute(0, 3, 1, 2).float()
    except Exception:
        return None


def write_csv(path: Path, header: list[str], rows: list[list[object]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def evaluate_clip_with_without(concepts: list[Concept], output_root: Path) -> Path:
    rows: list[list[object]] = []
    for c in concepts:
        folders = [
            output_root / f"output_a_photo_of_{c.safe_name}_with_baseline",
            output_root / f"output_a_photo_of_{c.safe_name}_without_finetuning",
        ]
        for folder in folders:
            if not folder.exists():
                continue
            images, prompts = load_images_for_clip(folder, c.visual_description)
            if not images:
                continue
            scores = calculate_clip_scores(images, prompts)
            rows.append([str(folder), float(np.mean(scores)), float(np.std(scores))])

    out_csv = output_root / "clip_score_baseline.csv"
    write_csv(out_csv, ["Folder", "Average CLIP Score", "Standard Deviation"], rows)
    return out_csv


def evaluate_fid_with_without(concepts: list[Concept], output_root: Path, fid_resize: int, num_runs: int) -> Path:
    rows: list[list[object]] = []
    for c in concepts:
        ref_files = [x for x in c.folder.iterdir() if x.is_file() and is_image_file(x)]
        ref_images = [load_and_process_for_fid(x, fid_resize) for x in ref_files]
        ref_images = [x for x in ref_images if x is not None]
        if not ref_images:
            continue
        ref_tensor = torch.cat(ref_images)

        generated_folders = [
            output_root / f"output_a_photo_of_{c.safe_name}_with_baseline",
            output_root / f"output_a_photo_of_{c.safe_name}_without_finetuning",
        ]
        for folder in generated_folders:
            if not folder.exists():
                continue
            gen_files = [x for x in folder.iterdir() if x.is_file() and is_image_file(x)]
            gen_images = [load_and_process_for_fid(x, fid_resize) for x in gen_files]
            gen_images = [x for x in gen_images if x is not None]
            if not gen_images:
                continue
            gen_tensor = torch.cat(gen_images)

            fid_values = []
            for _ in range(num_runs):
                metric = FrechetInceptionDistance(normalize=True)
                metric.update(ref_tensor, real=True)
                metric.update(gen_tensor, real=False)
                fid_values.append(metric.compute().item())
            rows.append([str(folder), float(np.mean(fid_values)), float(np.std(fid_values))])

    out_csv = output_root / "fid_scores_results_baseline.csv"
    write_csv(out_csv, ["Folder", "FID Mean", "FID Std Dev"], rows)
    return out_csv


def evaluate_clip_checkpoints(concepts: list[Concept], output_root: Path, checkpoints: list[int]) -> Path:
    rows: list[list[object]] = []
    for c in concepts:
        for ckpt in checkpoints:
            folder = output_root / f"output_{c.safe_name}_checkpoint_{ckpt}"
            if not folder.exists():
                continue
            images, prompts = load_images_for_clip(folder, c.visual_description)
            if not images:
                continue
            scores = calculate_clip_scores(images, prompts)
            rows.append([str(folder), float(np.mean(scores)), float(np.std(scores))])
    out_csv = output_root / "clip_score_checkpoints_baseline.csv"
    write_csv(out_csv, ["Folder", "Average CLIP Score", "Standard Deviation"], rows)
    return out_csv


def evaluate_fid_checkpoints(
    concepts: list[Concept],
    output_root: Path,
    checkpoints: list[int],
    fid_resize: int,
    num_runs: int,
) -> Path:
    rows: list[list[object]] = []
    for c in concepts:
        ref_files = [x for x in c.folder.iterdir() if x.is_file() and is_image_file(x)]
        ref_images = [load_and_process_for_fid(x, fid_resize) for x in ref_files]
        ref_images = [x for x in ref_images if x is not None]
        if not ref_images:
            continue
        ref_tensor = torch.cat(ref_images)

        for ckpt in checkpoints:
            folder = output_root / f"output_{c.safe_name}_checkpoint_{ckpt}"
            if not folder.exists():
                continue
            gen_files = [x for x in folder.iterdir() if x.is_file() and is_image_file(x)]
            gen_images = [load_and_process_for_fid(x, fid_resize) for x in gen_files]
            gen_images = [x for x in gen_images if x is not None]
            if not gen_images:
                continue
            gen_tensor = torch.cat(gen_images)

            fid_values = []
            for _ in range(num_runs):
                metric = FrechetInceptionDistance(normalize=True)
                metric.update(ref_tensor, real=True)
                metric.update(gen_tensor, real=False)
                fid_values.append(metric.compute().item())

            rows.append([str(folder), float(np.mean(fid_values)), float(np.std(fid_values))])

    out_csv = output_root / "fid_scores_results_checkpoints_baseline.csv"
    write_csv(out_csv, ["Folder", "FID Mean", "FID Std Dev"], rows)
    return out_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Textual Inversion baseline experiments.")
    parser.add_argument("--data-root", type=Path, default=Path.cwd(), help="Root folder with concept subfolders.")
    parser.add_argument("--output-root", type=Path, default=Path.cwd() / "baseline_outputs")
    parser.add_argument(
        "--train-script",
        type=Path,
        default=Path("/workspace/diffusers/examples/textual_inversion/textual_inversion_sdxl.py"),
        help="Path to diffusers textual inversion SDXL training script.",
    )
    parser.add_argument("--concepts", nargs="*", default=None, help="Optional subset of concept folder names.")
    parser.add_argument("--base-model", default="stabilityai/stable-diffusion-xl-base-1.0")
    parser.add_argument("--vae-path", default="madebyollin/sdxl-vae-fp16-fix")
    parser.add_argument("--initializer-token", default="object")
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--max-train-steps", type=int, default=3000)
    parser.add_argument("--checkpointing-steps", type=int, default=500)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--mixed-precision", default="fp16")
    parser.add_argument("--num-images", type=int, default=100)
    parser.add_argument("--num-inference-steps", type=int, default=4)
    parser.add_argument("--guidance-scale", type=float, default=8.0)
    parser.add_argument("--fid-resize", type=int, default=256)
    parser.add_argument("--fid-runs", type=int, default=1)
    parser.add_argument("--checkpoints", nargs="*", type=int, default=[500, 1000, 1500, 2000, 2500, 3000])
    parser.add_argument("--disable-lcm", action="store_true", help="Disable LCM UNet for image generation.")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-generate", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument(
        "--extra-train-arg",
        action="append",
        default=[],
        help="Extra arg passed to train script (repeatable), e.g. --extra-train-arg=--use_8bit_adam",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this script.")

    concepts = discover_concepts(args.data_root, args.concepts)
    if not concepts:
        raise RuntimeError(f"No concept folders with images found in {args.data_root}")

    args.output_root.mkdir(parents=True, exist_ok=True)

    print("Concepts:", [c.name for c in concepts])
    train_dirs: dict[str, Path] = {c.safe_name: args.output_root / "training_runs" / c.safe_name for c in concepts}

    if not args.skip_train:
        if not args.train_script.exists():
            raise FileNotFoundError(f"Training script not found: {args.train_script}")
        for concept in concepts:
            out_dir = train_textual_inversion(
                concept=concept,
                output_root=args.output_root,
                train_script=args.train_script,
                model_name=args.base_model,
                vae_path=args.vae_path,
                resolution=args.resolution,
                train_batch_size=args.train_batch_size,
                grad_accum=args.gradient_accumulation_steps,
                lr=args.learning_rate,
                max_train_steps=args.max_train_steps,
                checkpoint_steps=args.checkpointing_steps,
                save_steps=args.save_steps,
                mixed_precision=args.mixed_precision,
                initializer_token=args.initializer_token,
                extra_args=args.extra_train_arg,
            )
            train_dirs[concept.safe_name] = out_dir

    if not args.skip_generate:
        pipe = load_pipeline(args.base_model, use_lcm=not args.disable_lcm)
        for concept in concepts:
            concept_train_dir = train_dirs[concept.safe_name]
            final_embedding = find_final_embedding(concept_train_dir)

            without_dir = args.output_root / f"output_a_photo_of_{concept.safe_name}_without_finetuning"
            generate_images(
                pipe=pipe,
                prompt=concept.prompt_base,
                out_dir=without_dir,
                n_images=args.num_images,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
            )

            pipe.load_textual_inversion(str(final_embedding), token=concept.token)
            with_dir = args.output_root / f"output_a_photo_of_{concept.safe_name}_with_baseline"
            generate_images(
                pipe=pipe,
                prompt=concept.prompt_ti,
                out_dir=with_dir,
                n_images=args.num_images,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
            )

            for ckpt in args.checkpoints:
                ckpt_embedding = find_checkpoint_embedding(concept_train_dir, ckpt)
                if ckpt_embedding is None:
                    continue
                pipe.load_textual_inversion(str(ckpt_embedding), token=concept.token)
                ckpt_dir = args.output_root / f"output_{concept.safe_name}_checkpoint_{ckpt}"
                generate_images(
                    pipe=pipe,
                    prompt=concept.prompt_ti,
                    out_dir=ckpt_dir,
                    n_images=args.num_images,
                    num_inference_steps=args.num_inference_steps,
                    guidance_scale=args.guidance_scale,
                )
            unload_ti_if_supported(pipe, concept.token)

    if not args.skip_eval:
        clip_main = evaluate_clip_with_without(concepts, args.output_root)
        fid_main = evaluate_fid_with_without(concepts, args.output_root, args.fid_resize, args.fid_runs)
        clip_ckpt = evaluate_clip_checkpoints(concepts, args.output_root, args.checkpoints)
        fid_ckpt = evaluate_fid_checkpoints(concepts, args.output_root, args.checkpoints, args.fid_resize, args.fid_runs)
        print("Saved:", clip_main)
        print("Saved:", fid_main)
        print("Saved:", clip_ckpt)
        print("Saved:", fid_ckpt)


if __name__ == "__main__":
    main()
