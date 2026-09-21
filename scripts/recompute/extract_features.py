"""Download the generated-image tars from HF one at a time, extract InceptionV3 features
(same preprocessing as experiment_utils: resize to 256, uint8, torchmetrics NoTrainInceptionV3),
cache them as .npy, and delete the tar. Resumable: folders whose .npy exists are skipped.
"""
import io
import os
import sys
import tarfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.multimodal.clip_score import CLIPScore
from huggingface_hub import hf_hub_download

REPO = "FelipeMahlow/auto-tune-generated-images"
REPO_ROOT = Path(__file__).resolve().parents[2]
WORK = Path(os.environ.get("RECOMPUTE_WORK", REPO_ROOT / "scripts" / "recompute" / "work"))
RECOMP_DIR = REPO_ROOT / "results" / "recomputed"
FEAT = WORK / "features"
CACHE = WORK / "hf_cache"
FEAT.mkdir(parents=True, exist_ok=True)
CACHE.mkdir(parents=True, exist_ok=True)
RESIZE = 256
BATCH = 32
IMG_EXT = {".png", ".jpg", ".jpeg", ".webp"}

# CLIP needs the images themselves, and the Textual Inversion folders published on HF are the
# safety-filtered ones, so their CLIP scores have to be recomputed here (DreamBooth's published
# images are the same ones already scored, so those are reused from the CSVs).
VISUAL_DESCRIPTIONS = {
    "patua": "A small amulet or charm, often worn as a necklace, made of fabric or leather.",
    "cuscuz": "A plate of golden, grainy steamed cornmeal, often served with cheese or meat.",
    "chaneques": "Mythical small humanoid creatures, resembling goblins, from Mexican folklore.",
    "chamanto": "A traditional Chilean poncho with intricate patterns, worn over the shoulders.",
    "lokum": "Turkish delight, a gelatinous sweet dusted with powdered sugar, in various colors.",
    "pacoca": (
        "A cylindrical or rectangular Brazilian sweet made of ground peanuts, sugar, and salt, "
        "with a crumbly and slightly rough texture, typically light brown in color."
    ),
    "jian": "A straight double-edged Chinese sword with a narrow, elegant blade.",
    "saci": "A one-legged trickster from Brazilian folklore, with dark skin, wearing a red cap and smoking a pipe.",
}

CONCEPTS = ["chamanto", "chaneques", "cuscuz", "jian", "lokum", "pacoca", "patua", "saci"]
REAL_DIRS = {
    "chamanto": "chamanto", "chaneques": "chaneques", "cuscuz": "cuscuz", "jian": "jian",
    "lokum": "lokum", "pacoca": "paçoca", "patua": "patuá", "saci": "saci",
}
# the scraped training images are not in git; point REAL_IMAGES_ROOT at the folder holding
# chamanto/, chaneques/, ..., saci/ (20 images each)
REAL_ROOT = Path(os.environ.get("REAL_IMAGES_ROOT", REPO_ROOT))

device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
# the metric itself keeps float64 buffers (unsupported on MPS); only the network needs the GPU
_inception = FrechetInceptionDistance(feature=2048, normalize=False).inception.to(device)
_inception.eval()
_clip = None


def clip_metric():
    global _clip
    if _clip is None:
        _clip = CLIPScore(model_name_or_path="openai/clip-vit-base-patch16").to(device)
        _clip.eval()
    return _clip


def clip_score_of(img: Image.Image, prompt_text: str) -> float:
    """One image at a time, as compute_clip_statistics does, on the full-size image."""
    arr = np.asarray(img.convert("RGB"), dtype=np.uint8)
    tensor = torch.from_numpy(arr.copy()).permute(2, 0, 1).unsqueeze(0).to(device)
    with torch.no_grad():
        return float(clip_metric()(tensor, [prompt_text]).detach().cpu())


def features_from_batches(images):
    """images: iterable of HxWx3 uint8 numpy arrays."""
    out = []
    buf = []
    with torch.no_grad():
        for arr in images:
            buf.append(torch.from_numpy(arr).permute(2, 0, 1))
            if len(buf) == BATCH:
                out.append(_inception(torch.stack(buf).to(device)).cpu().double())
                buf = []
        if buf:
            out.append(_inception(torch.stack(buf).to(device)).cpu().double())
    return torch.cat(out).numpy()


def prep(img: Image.Image) -> np.ndarray:
    return np.asarray(img.convert("RGB").resize((RESIZE, RESIZE), Image.BILINEAR), dtype=np.uint8)


def folder_of(member_name: str) -> str:
    parts = Path(member_name).parts
    return parts[-2] if len(parts) >= 2 else ""


def process_tar(tar_path: Path, mapping: dict, clip_prompt: str | None = None) -> None:
    """mapping: folder name inside the tar -> output feature file stem.
    With clip_prompt set, per-image CLIP scores are also computed and saved as <stem>_clip.npy."""
    buckets = {w: [] for w in mapping}
    clip_scores = {w: [] for w in mapping}
    with tarfile.open(tar_path, "r|*") as tf:  # streaming
        for member in tf:
            if not member.isfile():
                continue
            if Path(member.name).suffix.lower() not in IMG_EXT:
                continue
            folder = folder_of(member.name)
            if folder not in buckets:
                continue
            data = tf.extractfile(member).read()
            image = Image.open(io.BytesIO(data))
            buckets[folder].append(prep(image))
            if clip_prompt is not None:
                clip_scores[folder].append(clip_score_of(image, clip_prompt))
    for folder, imgs in buckets.items():
        if not imgs:
            print(f"  !! no images found for {folder}", flush=True)
            continue
        feats = features_from_batches(imgs)
        stem = mapping[folder]
        np.save(FEAT / f"{stem}.npy", feats)
        if clip_prompt is not None and clip_scores[folder]:
            np.save(FEAT / f"{stem}_clip.npy", np.asarray(clip_scores[folder], dtype=np.float64))
        print(f"  saved {stem}: {feats.shape}", flush=True)


def wanted_for(concept: str, kind: str) -> dict:
    """folder inside the tar -> feature file stem."""
    if kind == "dreambooth":
        names = {
            f"output_a_photo_of_{concept}_without_finetuning": f"output_a_photo_of_{concept}_without_finetuning",
            f"output_a_photo_of_{concept}_with_finetuning": f"output_a_photo_of_{concept}_with_finetuning",
        }
        names.update({f"output_{concept}_checkpoint_{s}": f"output_{concept}_checkpoint_{s}"
                      for s in (500, 1000, 1500, 2000, 2500)})
        return names
    names = {f"output_a_photo_of_{concept}_with_baseline": f"output_a_photo_of_{concept}_with_baseline"}
    if concept == "saci":  # TI generated its own base-model images for saci
        names[f"output_a_photo_of_{concept}_without_finetuning"] = f"output_a_photo_of_{concept}_without_finetuning_ti"
    return names


def main():
    # real training images (local, never uploaded)
    for concept, dirname in REAL_DIRS.items():
        out = FEAT / f"real_{concept}.npy"
        if out.exists():
            continue
        folder = REAL_ROOT / dirname
        files = sorted(p for p in folder.iterdir() if p.suffix.lower() in IMG_EXT)
        feats = features_from_batches(prep(Image.open(p)) for p in files)
        np.save(out, feats)
        print(f"real {concept}: {feats.shape}", flush=True)

    for concept in CONCEPTS:
        for kind, fname in (("dreambooth", f"dreambooth_{concept}.tar"),
                            ("ti", f"textual_inversion_{concept}.tar")):
            mapping = wanted_for(concept, kind)
            missing = {folder: stem for folder, stem in mapping.items()
                       if not (FEAT / f"{stem}.npy").exists()
                       or (kind == "ti" and not (FEAT / f"{stem}_clip.npy").exists())}
            if not missing:
                print(f"skip {fname} (cached)", flush=True)
                continue
            print(f"downloading {fname} ...", flush=True)
            path = Path(hf_hub_download(REPO, fname, repo_type="dataset", cache_dir=CACHE))
            print(f"  processing {fname} ({path.stat().st_size/1e9:.1f} GB)", flush=True)
            process_tar(path, missing, VISUAL_DESCRIPTIONS[concept] if kind == "ti" else None)
            # free the ~3 GB blob (and its symlink target) before the next download
            try:
                real = path.resolve()
                path.unlink(missing_ok=True)
                real.unlink(missing_ok=True)
            except OSError as exc:
                print(f"  could not delete {path}: {exc}", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    sys.exit(main())
