"""Rebuild Figure 3: CLIP Score, FID and KID across training steps.

FID and KID are computed on 100 images at every point, including step 0 and step 3000 (where a
100-image subsample is drawn from the 1000 available, averaged over 50 draws), because both metrics
depend on the sample size and the old figure mixed 100-image checkpoints with 1000-image endpoints.
CLIP Score is a per-image mean, so it is unaffected and uses all available images.
"""
import csv
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
WORK = Path(os.environ.get("RECOMPUTE_WORK", REPO_ROOT / "scripts" / "recompute" / "work"))
RECOMP_DIR = REPO_ROOT / "results" / "recomputed"
REPO = REPO_ROOT
RECOMP = RECOMP_DIR / "metrics_recomputed.csv"
STEPS = [0, 500, 1000, 1500, 2000, 2500, 3000]
LABELS = {"lokum": "Lokum", "pacoca": "Paçoca", "cuscuz": "Cuscuz", "saci": "Saci",
          "chaneques": "Chaneques", "jian": "Jian", "chamanto": "Chamanto", "patua": "Patuá"}
ORDER = ["lokum", "pacoca", "cuscuz", "saci", "chaneques", "jian", "chamanto", "patua"]
COLORS = dict(zip(ORDER, ["#1f77b4", "#ff7f0e", "#2ca02c", "#e6b800", "#e377c2",
                          "#8c564b", "#4b0082", "#d62728"]))


def read(path):
    with open(path) as fh:
        return list(csv.DictReader(fh))


def main():
    clip = {c: {} for c in ORDER}
    for row in read(REPO / "results/dreambooth/metrics_dreambooth.csv"):
        c = row["concept_safe"]
        if row["condition"] == "without_finetuning":
            clip[c][0] = float(row["clip_mean"])
        elif row["condition"] == "with_finetuning":
            clip[c][3000] = float(row["clip_mean"])
    for row in read(REPO / "results/dreambooth/metrics_checkpoints_dreambooth.csv"):
        step = int(row["condition"].rsplit("_", 1)[1])
        if step != 3000:
            clip[row["concept_safe"]][step] = float(row["clip_mean"])

    fid = {c: {} for c in ORDER}
    kid = {c: {} for c in ORDER}
    for row in read(RECOMP):
        c, cond = row["concept"], row["condition"]
        if row["method"] != "dreambooth_lora":
            continue
        if cond == "without_finetuning":
            step = 0
        elif cond == "with_finetuning":
            step = 3000
        elif cond.startswith("checkpoint_"):
            step = int(cond.rsplit("_", 1)[1])
        else:
            continue
        # endpoints: the 100-image subsample average; checkpoints already have 100 images
        fid[c][step] = float(row["fid_sub100"] or 0) if step in (0, 3000) else float(row["fid"])
        kid[c][step] = float(row["kid_sub100"] or 0) if step in (0, 3000) else float(row["kid"])

    # Keep the PDF close to its final width in the manuscript, so font sizes
    # remain readable after LaTeX scales it to \linewidth.
    plt.rcParams.update({
        "font.size": 9,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
    })
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 3.5))
    for ax, data, title, ylabel in (
        (axes[0], clip, "(a) CLIP Score", "CLIP Score"),
        (axes[1], fid, "(b) FID\n(100 images per point)", "FID"),
        (axes[2], kid, "(c) KID\n(100 images per point)", "KID"),
    ):
        for c in ORDER:
            xs = [s for s in STEPS if s in data[c]]
            ax.plot(xs, [data[c][s] for s in xs], marker="o", markersize=3,
                    color=COLORS[c], label=LABELS[c], linewidth=1.2)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Training steps")
        ax.set_ylabel(ylabel)
        ax.set_xticks(STEPS[::2])
        ax.grid(alpha=0.3)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, 0.005))
    fig.tight_layout(rect=(0, 0.12, 1, 1), w_pad=1.6)
    out = REPO / "paper_tex" / "fig_3_paper.pdf"
    fig.savefig(out, bbox_inches="tight")
    WORK.mkdir(parents=True, exist_ok=True)
    fig.savefig(WORK / "fig_3_preview.png", dpi=110, bbox_inches="tight")
    print("wrote", out)

    for name, data in (("CLIP", clip), ("FID", fid), ("KID", kid)):
        best = {c: max(data[c], key=data[c].get) if name == "CLIP" else min(data[c], key=data[c].get)
                for c in ORDER}
        print(name, "best step per concept:", best)


if __name__ == "__main__":
    main()
