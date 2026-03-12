#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

import torch

from experiment_utils import (
    Concept,
    ProgressTracker,
    aggregate_metrics_by_group,
    create_human_eval_package,
    create_qualitative_sheet,
    discover_concepts,
    evaluate_generated_folder,
    write_csv,
)


def run_cmd(cmd: list[str]) -> None:
    print(" ".join(str(part) for part in cmd))
    subprocess.run(cmd, check=True)


def resolve_train_script(user_path: Path | None, relative_script: str) -> Path:
    if user_path is not None:
        if user_path.exists():
            return user_path
        raise FileNotFoundError(f"Training script not found: {user_path}")

    repo_candidates = []
    env_repo = os.environ.get("DIFFUSERS_REPO")
    if env_repo:
        repo_candidates.append(Path(env_repo))
    repo_candidates.extend(
        [
            Path.cwd() / "diffusers",
            Path("/workspace/diffusers"),
            Path("/workspace") / "diffusers",
            Path.home() / "diffusers",
        ]
    )

    for repo in repo_candidates:
        candidate = repo / relative_script
        if candidate.exists():
            return candidate

    searched = ", ".join(str(repo / relative_script) for repo in repo_candidates)
    raise FileNotFoundError(
        "Could not locate the diffusers training script. "
        f"Pass --train-script explicitly or set DIFFUSERS_REPO. Searched: {searched}"
    )


def import_diffusers_components():
    try:
        from diffusers import DiffusionPipeline, LCMScheduler, UNet2DConditionModel
    except Exception as exc:
        raise RuntimeError(
            "Failed to import diffusers. This usually means the installed diffusers/torch "
            "versions are incompatible in the current environment. Try pinning to a known "
            "working pair before running generation."
        ) from exc
    return DiffusionPipeline, LCMScheduler, UNet2DConditionModel


def load_pipeline(base_model: str, use_lcm: bool):
    DiffusionPipeline, LCMScheduler, UNet2DConditionModel = import_diffusers_components()
    if use_lcm:
        unet = UNet2DConditionModel.from_pretrained(
            "latent-consistency/lcm-sdxl",
            torch_dtype=torch.float16,
            variant="fp16",
        )
        pipe = DiffusionPipeline.from_pretrained(base_model, unet=unet, torch_dtype=torch.float16).to("cuda")
        pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
        return pipe
    return DiffusionPipeline.from_pretrained(base_model, torch_dtype=torch.float16).to("cuda")


def unload_textual_inversion(pipe, token: str) -> None:
    if hasattr(pipe, "unload_textual_inversion"):
        try:
            pipe.unload_textual_inversion(token)
        except Exception:
            pass


def generate_images(
    pipe,
    prompt: str,
    out_dir: Path,
    n_images: int,
    num_inference_steps: int,
    guidance_scale: float,
    seed_start: int,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict[str, object]] = []
    for image_index in range(n_images):
        generator = torch.Generator(device="cuda").manual_seed(seed_start + image_index)
        image = pipe(
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        ).images[0]
        image_path = out_dir / f"image_{image_index + 1:04d}.png"
        image.save(image_path)
        manifest_rows.append({"image_path": str(image_path), "seed": seed_start + image_index, "prompt": prompt})
    write_csv(out_dir / "generation_manifest.csv", ["image_path", "seed", "prompt"], manifest_rows)


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
    enable_xformers: bool,
    extra_args: list[str],
) -> tuple[Path, str]:
    concept_output = output_root / "training_runs" / concept.safe_name
    concept_output.mkdir(parents=True, exist_ok=True)
    token = f"<ti_{concept.safe_name}>"
    cmd = [
        "accelerate",
        "launch",
        str(train_script),
        f"--pretrained_model_name_or_path={model_name}",
        f"--pretrained_vae_model_name_or_path={vae_path}",
        f"--train_data_dir={concept.folder}",
        f"--output_dir={concept_output}",
        f"--placeholder_token={token}",
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
        f"--validation_prompt=a photo of {token}",
        "--validation_epochs=25",
    ]
    if enable_xformers:
        cmd.append("--enable_xformers_memory_efficient_attention")
    if mixed_precision:
        cmd.append(f"--mixed_precision={mixed_precision}")
    cmd.extend(extra_args)
    run_cmd(cmd)
    return concept_output, token


def find_textual_inversion_embedding(concept_train_dir: Path, step: int | None = None) -> Path:
    patterns = []
    if step is None:
        patterns.extend(["learned_embeds.safetensors", "learned_embeds.bin", "learned_embeds.pt"])
    else:
        patterns.extend(
            [
                f"learned_embeds-steps-{step}.safetensors",
                f"learned_embeds-steps-{step}.bin",
                f"learned_embeds-steps-{step}.pt",
                f"checkpoint-{step}/learned_embeds.safetensors",
                f"checkpoint-{step}/learned_embeds.bin",
                f"checkpoint-{step}/learned_embeds.pt",
            ]
        )
    for pattern in patterns:
        candidate = concept_train_dir / pattern
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Embedding not found in {concept_train_dir} for step={step}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Textual Inversion baseline experiments.")
    parser.add_argument("--data-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-root", type=Path, default=Path.cwd() / "baseline_outputs")
    parser.add_argument(
        "--train-script",
        type=Path,
        default=None,
    )
    parser.add_argument("--concepts", nargs="*", default=None)
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
    parser.add_argument("--enable-xformers", action="store_true")
    parser.add_argument("--num-images", type=int, default=1000)
    parser.add_argument("--num-inference-steps", type=int, default=4)
    parser.add_argument("--guidance-scale", type=float, default=8.0)
    parser.add_argument("--bootstrap-samples", type=int, default=200)
    parser.add_argument("--metric-seed", type=int, default=42)
    parser.add_argument("--seed-start", type=int, default=1000)
    parser.add_argument("--kid-subset-size", type=int, default=50)
    parser.add_argument("--fid-resize", type=int, default=256)
    parser.add_argument("--qualitative-samples", type=int, default=6)
    parser.add_argument("--human-eval-samples", type=int, default=25)
    parser.add_argument("--checkpoints", nargs="*", type=int, default=[500, 1000, 1500, 2000, 2500, 3000])
    parser.add_argument("--disable-lcm", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-generate", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--extra-train-arg", action="append", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this script.")

    concepts = discover_concepts(args.data_root, args.concepts)
    if not concepts:
        raise RuntimeError(f"No concept folders with images found in {args.data_root}")

    args.output_root.mkdir(parents=True, exist_ok=True)
    total_steps = 0
    if not args.skip_train:
        total_steps += len(concepts)
    if not args.skip_generate:
        total_steps += len(concepts) * (2 + len(args.checkpoints))
    if not args.skip_eval:
        total_steps += len(concepts) * (2 + len(args.checkpoints))
        total_steps += 4
    progress = ProgressTracker(args.output_root, "textual_inversion_baseline", total_steps)
    progress.log(f"Starting run for concepts: {[concept.name for concept in concepts]}")
    train_dirs = {concept.safe_name: args.output_root / "training_runs" / concept.safe_name for concept in concepts}
    learned_tokens = {concept.safe_name: f"<ti_{concept.safe_name}>" for concept in concepts}

    if not args.skip_train:
        train_script = resolve_train_script(
            args.train_script, "examples/textual_inversion/textual_inversion_sdxl.py"
        )
        progress.log(f"Using training script: {train_script}")
        for concept in concepts:
            progress.set_stage("training", concept.name, "starting")
            progress.log(f"Training {concept.name}")
            concept_output, token = train_textual_inversion(
                concept=concept,
                output_root=args.output_root,
                train_script=train_script,
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
                enable_xformers=args.enable_xformers,
                extra_args=args.extra_train_arg,
            )
            train_dirs[concept.safe_name] = concept_output
            learned_tokens[concept.safe_name] = token
            progress.advance(stage="training", concept=concept.name, detail="completed")

    if not args.skip_generate:
        progress.set_stage("generation", detail="loading pipeline")
        pipe = load_pipeline(args.base_model, use_lcm=not args.disable_lcm)
        for concept_index, concept in enumerate(concepts):
            concept_seed = args.seed_start + concept_index * (args.num_images * 10)
            without_dir = args.output_root / f"output_a_photo_of_{concept.safe_name}_without_finetuning"
            progress.set_stage("generation", concept.name, "without_finetuning")
            generate_images(
                pipe=pipe,
                prompt=concept.prompt_base,
                out_dir=without_dir,
                n_images=args.num_images,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                seed_start=concept_seed,
            )
            progress.advance(stage="generation", concept=concept.name, detail="without_finetuning_done")

            final_embedding = find_textual_inversion_embedding(train_dirs[concept.safe_name], None)
            token = learned_tokens[concept.safe_name]
            pipe.load_textual_inversion(str(final_embedding), token=token)
            with_dir = args.output_root / f"output_a_photo_of_{concept.safe_name}_with_baseline"
            progress.set_stage("generation", concept.name, "with_baseline")
            generate_images(
                pipe=pipe,
                prompt=f"a photo of {token}",
                out_dir=with_dir,
                n_images=args.num_images,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                seed_start=concept_seed,
            )
            unload_textual_inversion(pipe, token)
            progress.advance(stage="generation", concept=concept.name, detail="with_baseline_done")

            for checkpoint in args.checkpoints:
                try:
                    checkpoint_embedding = find_textual_inversion_embedding(train_dirs[concept.safe_name], checkpoint)
                except FileNotFoundError:
                    progress.advance(stage="generation", concept=concept.name, detail=f"checkpoint_{checkpoint}_missing")
                    continue
                pipe.load_textual_inversion(str(checkpoint_embedding), token=token)
                checkpoint_dir = args.output_root / f"output_{concept.safe_name}_checkpoint_{checkpoint}"
                progress.set_stage("generation", concept.name, f"checkpoint_{checkpoint}")
                generate_images(
                    pipe=pipe,
                    prompt=f"a photo of {token}",
                    out_dir=checkpoint_dir,
                    n_images=args.num_images,
                    num_inference_steps=args.num_inference_steps,
                    guidance_scale=args.guidance_scale,
                    seed_start=concept_seed,
                )
                unload_textual_inversion(pipe, token)
                progress.advance(stage="generation", concept=concept.name, detail=f"checkpoint_{checkpoint}_done")

    if args.skip_eval:
        progress.complete()
        return

    main_rows: list[dict[str, object]] = []
    checkpoint_rows: list[dict[str, object]] = []
    human_eval_conditions: dict[str, list[tuple[str, Path]]] = {}
    qualitative_root = args.output_root / "qualitative"

    for concept in concepts:
        without_dir = args.output_root / f"output_a_photo_of_{concept.safe_name}_without_finetuning"
        with_dir = args.output_root / f"output_a_photo_of_{concept.safe_name}_with_baseline"
        condition_folders = []

        if without_dir.exists():
            progress.set_stage("evaluation", concept.name, "without_finetuning")
            main_rows.append(
                evaluate_generated_folder(
                    concept=concept,
                    generated_folder=without_dir,
                    condition_label="without_finetuning",
                    method_label="base_model",
                    bootstrap_samples=args.bootstrap_samples,
                    metric_seed=args.metric_seed,
                    fid_resize=args.fid_resize,
                    kid_subset_size=args.kid_subset_size,
                )
            )
            condition_folders.append(("without_finetuning", without_dir))
            progress.advance(stage="evaluation", concept=concept.name, detail="without_finetuning_done")

        if with_dir.exists():
            progress.set_stage("evaluation", concept.name, "with_baseline")
            main_rows.append(
                evaluate_generated_folder(
                    concept=concept,
                    generated_folder=with_dir,
                    condition_label="with_baseline",
                    method_label="textual_inversion",
                    bootstrap_samples=args.bootstrap_samples,
                    metric_seed=args.metric_seed + 10,
                    fid_resize=args.fid_resize,
                    kid_subset_size=args.kid_subset_size,
                )
            )
            condition_folders.append(("with_baseline", with_dir))
            progress.advance(stage="evaluation", concept=concept.name, detail="with_baseline_done")

        for checkpoint in args.checkpoints:
            checkpoint_dir = args.output_root / f"output_{concept.safe_name}_checkpoint_{checkpoint}"
            if not checkpoint_dir.exists():
                progress.advance(stage="evaluation", concept=concept.name, detail=f"checkpoint_{checkpoint}_missing")
                continue
            progress.set_stage("evaluation", concept.name, f"checkpoint_{checkpoint}")
            checkpoint_rows.append(
                evaluate_generated_folder(
                    concept=concept,
                    generated_folder=checkpoint_dir,
                    condition_label=f"checkpoint_{checkpoint}",
                    method_label="textual_inversion",
                    bootstrap_samples=args.bootstrap_samples,
                    metric_seed=args.metric_seed + checkpoint,
                    fid_resize=args.fid_resize,
                    kid_subset_size=args.kid_subset_size,
                )
            )
            progress.advance(stage="evaluation", concept=concept.name, detail=f"checkpoint_{checkpoint}_done")

        if condition_folders:
            create_qualitative_sheet(
                concept=concept,
                reference_folder=concept.folder,
                condition_folders=condition_folders,
                destination=qualitative_root / f"{concept.safe_name}_qualitative.png",
                samples_per_row=args.qualitative_samples,
                seed=args.metric_seed,
            )
            human_eval_conditions[concept.safe_name] = condition_folders

    metric_fields = [
        "concept",
        "concept_safe",
        "group",
        "method",
        "condition",
        "folder",
        "sample_count",
        "clip_mean",
        "clip_std",
        "clip_ci95_low",
        "clip_ci95_high",
        "fid_mean",
        "fid_std",
        "fid_ci95_low",
        "fid_ci95_high",
        "kid_mean",
        "kid_std",
        "kid_ci95_low",
        "kid_ci95_high",
    ]
    write_csv(args.output_root / "metrics_baseline.csv", metric_fields, main_rows)
    write_csv(args.output_root / "metrics_checkpoints_baseline.csv", metric_fields, checkpoint_rows)
    group_metric_fields = [
        "group",
        "method",
        "condition",
        "num_concepts",
        "total_sample_count",
        "clip_group_mean",
        "clip_group_std",
        "clip_group_ci95_low",
        "clip_group_ci95_high",
        "fid_group_mean",
        "fid_group_std",
        "fid_group_ci95_low",
        "fid_group_ci95_high",
        "kid_group_mean",
        "kid_group_std",
        "kid_group_ci95_low",
        "kid_group_ci95_high",
    ]
    write_csv(
        args.output_root / "metrics_groupwise_baseline.csv",
        group_metric_fields,
        aggregate_metrics_by_group(main_rows, args.bootstrap_samples, args.metric_seed + 100),
    )
    progress.advance(stage="evaluation", detail="groupwise_main_written")
    write_csv(
        args.output_root / "metrics_groupwise_checkpoints_baseline.csv",
        group_metric_fields,
        aggregate_metrics_by_group(checkpoint_rows, args.bootstrap_samples, args.metric_seed + 200),
    )
    progress.advance(stage="evaluation", detail="groupwise_checkpoints_written")
    create_human_eval_package(
        concepts=concepts,
        condition_map=human_eval_conditions,
        output_root=args.output_root,
        samples_per_condition=args.human_eval_samples,
        seed=args.metric_seed,
    )
    progress.advance(stage="evaluation", detail="human_eval_written")
    progress.advance(stage="evaluation", detail="metrics_written")
    progress.complete()


if __name__ == "__main__":
    main()
