"""Recompute FID/KID (and CLIP for Textual Inversion) from cached Inception features.

Two fixes over the original run:
  1. the bootstrap keeps the 20 real images fixed and resamples only the generated images, so the
     interval is centered on the point estimate instead of being pushed up by duplicated real images;
  2. FID/KID at step 0 and step 3000 are also computed on 100-image subsamples, so the training
     curve compares like with like (FID falls as the sample grows).

The FID bootstrap uses the fact that the real covariance has rank <= 19: the trace term only needs
the generated features projected onto that subspace, which turns a 2048x2048 eigendecomposition per
resample into a 19x19 one.
"""
import csv
import json
import os
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
WORK = Path(os.environ.get("RECOMPUTE_WORK", REPO_ROOT / "scripts" / "recompute" / "work"))
RECOMP_DIR = REPO_ROOT / "results" / "recomputed"
FEAT = WORK / "features"
OUT = RECOMP_DIR
OUT.mkdir(parents=True, exist_ok=True)

CONCEPTS = ["chamanto", "chaneques", "cuscuz", "jian", "lokum", "pacoca", "patua", "saci"]
GROUPS = {
    "cuscuz": "food", "lokum": "food", "pacoca": "food",
    "saci": "folklore_being", "chaneques": "folklore_being",
    "patua": "artifact_object_clothing", "jian": "artifact_object_clothing",
    "chamanto": "artifact_object_clothing",
}
CHECKPOINTS = [500, 1000, 1500, 2000, 2500]
B = 1000          # bootstrap resamples
SUBSET_DRAWS = 50  # draws used for the 100-image FID/KID of steps 0 and 3000
SEED = 20260917


def load(stem):
    path = FEAT / f"{stem}.npy"
    return np.load(path) if path.exists() else None


class RealRef:
    """Everything that only depends on the (fixed) real features."""

    def __init__(self, real):
        self.x = real
        self.m = real.shape[0]
        self.mu = real.mean(0)
        self.sigma = np.cov(real, rowvar=False, ddof=1)
        self.tr_sigma = np.trace(self.sigma)
        vals, vecs = np.linalg.eigh((self.sigma + self.sigma.T) / 2)
        keep = vals > max(vals.max(), 0) * 1e-12
        self.u = vecs[:, keep]                    # 2048 x k
        self.d = np.sqrt(np.clip(vals[keep], 0, None))  # k
        d = real.shape[1]
        kxx = (real @ real.T / d + 1.0) ** 3
        np.fill_diagonal(kxx, 0.0)
        self.kxx_term = kxx.sum() / (self.m * (self.m - 1))


def fid_exact(ref: RealRef, gen):
    mu_g = gen.mean(0)
    sigma_g = np.cov(gen, rowvar=False, ddof=1)
    diff = ref.mu - mu_g
    s = (ref.sigma + ref.sigma.T) / 2
    vals, vecs = np.linalg.eigh(s)
    s_sqrt = (vecs * np.sqrt(np.clip(vals, 0, None))) @ vecs.T
    b = s_sqrt @ sigma_g @ s_sqrt
    ev = np.linalg.eigvalsh((b + b.T) / 2)
    trace_sqrt = np.sqrt(np.clip(ev, 0, None)).sum()
    return float(diff @ diff + ref.tr_sigma + np.trace(sigma_g) - 2 * trace_sqrt)


def fid_from_stats(ref: RealRef, counts, x, y, sq, n):
    """FID for a weighted resample. counts: how many times each generated image was drawn.
    x: gen features (n x 2048), y: gen features projected on the real subspace (n x k),
    sq: per-image squared norms."""
    total = counts.sum()
    mu_g = (counts @ x) / total
    tr_sigma_g = (counts @ sq - total * (mu_g @ mu_g)) / (total - 1)
    mu_y = (counts @ y) / total
    yc = y - mu_y
    cov_y = (yc.T * counts) @ yc / (total - 1)
    m = (cov_y * ref.d).T * ref.d
    ev = np.linalg.eigvalsh((m + m.T) / 2)
    trace_sqrt = np.sqrt(np.clip(ev, 0, None)).sum()
    diff = ref.mu - mu_g
    return float(diff @ diff + ref.tr_sigma + tr_sigma_g - 2 * trace_sqrt)


def kid_from_counts(ref: RealRef, counts, kyy, kxy_colsum, kyy_diag):
    total = counts.sum()
    yy = (counts @ kyy @ counts - counts @ kyy_diag) / (total * (total - 1))
    xy = (counts @ kxy_colsum) / (ref.m * total)
    return float(ref.kxx_term + yy - 2 * xy)


def evaluate(ref: RealRef, gen, rng, bootstrap=True):
    d = gen.shape[1]
    n = gen.shape[0]
    y = gen @ ref.u
    sq = np.einsum("ij,ij->i", gen, gen)
    kyy = (gen @ gen.T / d + 1.0) ** 3
    kyy_diag = np.diag(kyy).copy()
    kxy_colsum = ((ref.x @ gen.T / d + 1.0) ** 3).sum(axis=0)
    ones = np.ones(n)

    fid = fid_exact(ref, gen)
    fid_proj = fid_from_stats(ref, ones, gen, y, sq, n)
    kid = kid_from_counts(ref, ones, kyy, kxy_colsum, kyy_diag)
    result = {"n": n, "fid": fid, "fid_projected": fid_proj, "kid": kid}
    if not bootstrap:
        return result
    fids, kids = np.empty(B), np.empty(B)
    for b in range(B):
        idx = rng.integers(0, n, size=n)
        counts = np.bincount(idx, minlength=n).astype(np.float64)
        fids[b] = fid_from_stats(ref, counts, gen, y, sq, n)
        kids[b] = kid_from_counts(ref, counts, kyy, kxy_colsum, kyy_diag)
    result.update(
        fid_ci_low=float(np.percentile(fids, 2.5)), fid_ci_high=float(np.percentile(fids, 97.5)),
        fid_boot_mean=float(fids.mean()), fid_std=float(fids.std(ddof=1)),
        kid_ci_low=float(np.percentile(kids, 2.5)), kid_ci_high=float(np.percentile(kids, 97.5)),
        kid_boot_mean=float(kids.mean()), kid_std=float(kids.std(ddof=1)),
    )
    return result


def subsampled(ref: RealRef, gen, rng, size=100, draws=SUBSET_DRAWS):
    """Mean FID/KID over `draws` subsets of `size` images, for curve comparability."""
    d = gen.shape[1]
    n = len(gen)
    y = gen @ ref.u
    sq = np.einsum("ij,ij->i", gen, gen)
    fids, kids = [], []
    for _ in range(draws):
        idx = rng.choice(n, size=size, replace=False)
        counts = np.zeros(n)
        counts[idx] = 1.0
        fids.append(fid_from_stats(ref, counts, gen, y, sq, n))
        sub = gen[idx]
        ksub = (sub @ sub.T / d + 1.0) ** 3
        np.fill_diagonal(ksub, 0.0)
        kxy = (ref.x @ sub.T / d + 1.0) ** 3
        kids.append(ref.kxx_term + ksub.sum() / (size * (size - 1)) - 2 * kxy.mean())
    return {"fid_sub100": float(np.mean(fids)), "fid_sub100_std": float(np.std(fids, ddof=1)),
            "kid_sub100": float(np.mean(kids)), "kid_sub100_std": float(np.std(kids, ddof=1))}


def bootstrap_mean(values, rng):
    values = np.asarray(values, dtype=np.float64)
    means = np.empty(B)
    for b in range(B):
        means[b] = rng.choice(values, size=values.size, replace=True).mean()
    return {"clip_mean": float(values.mean()), "clip_std": float(values.std(ddof=1)),
            "clip_ci_low": float(np.percentile(means, 2.5)),
            "clip_ci_high": float(np.percentile(means, 97.5))}


def main():
    rows = []
    for concept in CONCEPTS:
        real = load(f"real_{concept}")
        if real is None:
            continue
        ref = RealRef(real)
        rng = np.random.default_rng(SEED)
        conditions = [
            ("dreambooth_lora", "without_finetuning", f"output_a_photo_of_{concept}_without_finetuning"),
            ("dreambooth_lora", "with_finetuning", f"output_a_photo_of_{concept}_with_finetuning"),
            ("textual_inversion", "with_baseline", f"output_a_photo_of_{concept}_with_baseline"),
            ("textual_inversion", "without_finetuning_ti", f"output_a_photo_of_{concept}_without_finetuning_ti"),
        ] + [("dreambooth_lora", f"checkpoint_{s}", f"output_{concept}_checkpoint_{s}") for s in CHECKPOINTS]

        for method, condition, stem in conditions:
            gen = load(stem)
            if gen is None:
                continue
            row = {"concept": concept, "group": GROUPS[concept], "method": method,
                   "condition": condition, "stem": stem}
            row.update(evaluate(ref, gen, rng))
            if len(gen) > 200:  # 1000-image conditions also get the 100-image view
                row.update(subsampled(ref, gen, rng))
            clip = load(f"{stem}_clip")
            if clip is not None:
                row.update(bootstrap_mean(clip, rng))
            rows.append(row)
            print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v)
                              for k, v in row.items() if k != "stem"}), flush=True)

    if rows:
        keys = sorted({k for r in rows for k in r})
        with open(OUT / "metrics_recomputed.csv", "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=["concept", "group", "method", "condition", "n"]
                                    + [k for k in keys if k not in {"concept", "group", "method", "condition", "n", "stem"}])
            writer.writeheader()
            for r in rows:
                writer.writerow({k: v for k, v in r.items() if k != "stem"})
        print(f"wrote {OUT/'metrics_recomputed.csv'} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
