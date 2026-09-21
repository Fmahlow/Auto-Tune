"""Emit the LaTeX rows for Table 2 (per concept, before/after) and Table 3 (DreamBooth vs TI)
from the recomputed metrics, plus the group-level numbers quoted in the text."""
import csv
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
WORK = Path(os.environ.get("RECOMPUTE_WORK", REPO_ROOT / "scripts" / "recompute" / "work"))
RECOMP_DIR = REPO_ROOT / "results" / "recomputed"
REPO = REPO_ROOT
RECOMP = RECOMP_DIR / "metrics_recomputed.csv"
ORDER = ["lokum", "pacoca", "cuscuz", "saci", "chaneques", "jian", "chamanto", "patua"]
TEX = {"lokum": "Lokum", "pacoca": "Pa\\c{c}oca", "cuscuz": "Cuscuz", "saci": "Saci",
       "chaneques": "Chaneques", "jian": "Jian", "chamanto": "Chamanto", "patua": "Patu\\'a"}
GROUPS = {"food": ["cuscuz", "lokum", "pacoca"], "folklore_being": ["saci", "chaneques"],
          "artifact_object_clothing": ["jian", "chamanto", "patua"]}
GROUP_TEX = {"food": "Food", "folklore_being": "Folklore beings",
             "artifact_object_clothing": "Artifacts/clothing"}


def read(path):
    with open(path) as fh:
        return list(csv.DictReader(fh))


def main():
    rec = {}
    for row in read(RECOMP):
        rec[(row["concept"], row["condition"])] = row

    clip = {}
    for row in read(REPO / "results/dreambooth/metrics_dreambooth.csv"):
        clip[(row["concept_safe"], row["condition"])] = row

    print("% ---- Table 2 rows (DreamBooth; CLIP from the original run, FID/KID recomputed)")
    for c in ORDER:
        before, after = rec[(c, "without_finetuning")], rec[(c, "with_finetuning")]
        cb, ca = clip[(c, "without_finetuning")], clip[(c, "with_finetuning")]
        print(
            f"        \\textit{{{TEX[c]}}} & {float(cb['clip_mean']):.2f} & "
            f"{float(ca['clip_mean']):.2f} [{float(ca['clip_ci95_low']):.2f}, {float(ca['clip_ci95_high']):.2f}] & "
            f"{float(before['fid']):.1f} & "
            f"{float(after['fid']):.1f} [{float(after['fid_ci_low']):.1f}, {float(after['fid_ci_high']):.1f}] & "
            f"{float(before['kid']):.3f} & "
            f"{float(after['kid']):.3f} [{float(after['kid_ci_low']):.3f}, {float(after['kid_ci_high']):.3f}] \\\\"
        )

    print("\n% ---- Table 3 rows (DreamBooth vs Textual Inversion, safety-filtered TI sets)")
    for c in ORDER:
        db, ti = rec[(c, "with_finetuning")], rec.get((c, "with_baseline"))
        if ti is None:
            continue
        db_clip = float(clip[(c, "with_finetuning")]["clip_mean"])
        ti_clip = float(ti["clip_mean"])
        print(
            f"        \\textit{{{TEX[c]}}} & {db_clip:.2f} & {ti_clip:.2f} & {db_clip - ti_clip:+.2f} & "
            f"{float(db['fid']):.1f} & {float(ti['fid']):.1f} & {float(db['fid']) - float(ti['fid']):+.1f} & "
            f"{float(db['kid']):.3f} & {float(ti['kid']):.3f} & {float(db['kid']) - float(ti['kid']):+.3f} "
            f"& {int(float(ti['n']))} \\\\"
        )

    print("\n% ---- group means (unweighted over concepts) and ranges")
    for gname, members in GROUPS.items():
        for label, getter in (
            ("CLIP DB", lambda c: float(clip[(c, "with_finetuning")]["clip_mean"])),
            ("CLIP TI", lambda c: float(rec[(c, "with_baseline")]["clip_mean"])),
            ("FID DB", lambda c: float(rec[(c, "with_finetuning")]["fid"])),
            ("FID TI", lambda c: float(rec[(c, "with_baseline")]["fid"])),
            ("KID DB", lambda c: float(rec[(c, "with_finetuning")]["kid"])),
            ("KID TI", lambda c: float(rec[(c, "with_baseline")]["kid"])),
        ):
            vals = [getter(c) for c in members]
            fmt = "{:.3f}" if "KID" in label else ("{:.2f}" if "CLIP" in label else "{:.1f}")
            mean, lo, hi = sum(vals) / len(vals), min(vals), max(vals)
            print(f"{GROUP_TEX[gname]:<19} {label:<8} mean {fmt.format(mean)}  range [{fmt.format(lo)}, {fmt.format(hi)}]")

    print("\n% ---- old vs new (sanity): published point estimates should barely move")
    old = {(r["concept_safe"], r["condition"]): r for r in read(REPO / "results/dreambooth/metrics_dreambooth.csv")}
    for c in ORDER:
        o, n = old[(c, "with_finetuning")], rec[(c, "with_finetuning")]
        print(f"{c:<10} FID {float(o['fid_mean']):8.2f} -> {float(n['fid']):8.2f}   "
              f"KID {float(o['kid_mean']):.4f} -> {float(n['kid']):.4f}   "
              f"old CI [{float(o['fid_ci95_low']):.1f}, {float(o['fid_ci95_high']):.1f}] -> "
              f"new CI [{float(n['fid_ci_low']):.1f}, {float(n['fid_ci_high']):.1f}]")

    print("\n% ---- TI: filtered vs published unfiltered CLIP/FID/KID")
    old_ti = {r["concept_safe"]: r for r in read(REPO / "results/textual_inversion/metrics_baseline.csv")
              if r["condition"] == "with_baseline"}
    for c in ORDER:
        n = rec.get((c, "with_baseline"))
        o = old_ti.get(c)
        if not n or not o:
            continue
        print(f"{c:<10} n {int(float(n['n'])):4d}  CLIP {float(o['clip_mean']):.2f} -> {float(n['clip_mean']):.2f}   "
              f"FID {float(o['fid_mean']):.1f} -> {float(n['fid']):.1f}   "
              f"KID {float(o['kid_mean']):.3f} -> {float(n['kid']):.3f}")


if __name__ == "__main__":
    main()
