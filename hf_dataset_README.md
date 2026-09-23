---
license: mit
---

# Auto-Tune: Generated Images

Generated images supporting the paper *"Automated Fine-Tuning of Diffusion Models for
Learning Underrepresented Concepts from Web-Scraped Data"* (code at
https://github.com/Fmahlow/Auto-Tune; training images are not redistributed — they were
collected from Google Images and may be subject to third-party copyright; see the repository
README for instructions to re-collect them using the scraping scripts and concept names in
Table 1 of the paper).

Each `.tar` archive corresponds to one concept and one method:

- `dreambooth_<concept>.tar` — DreamBooth+LoRA outputs: `output_a_photo_of_<concept>_without_finetuning/`
  (1000 images, base SDXL model), `output_a_photo_of_<concept>_with_finetuning/` (1000 images,
  checkpoint-3000), and `output_<concept>_checkpoint_{500,1000,1500,2000,2500}/` (100 images each,
  intermediate checkpoints used for the training-curve figure).
- `textual_inversion_<concept>.tar` — Textual Inversion baseline: `output_a_photo_of_<concept>_with_baseline/`.
  The `without_finetuning` condition is identical to DreamBooth's (same base model, same
  seeds) and is not duplicated here, except for `saci`, whose `without_finetuning` folder is included since
  it was generated independently.
- `dreambooth_checkpoints_<concept>.tar` / `textual_inversion_checkpoints_<concept>.tar` — the
  fine-tuned LoRA weights (DreamBooth) and learned token embeddings (Textual Inversion), plus
  optimizer/scheduler state, at each saved checkpoint.

Concepts: chamanto, chaneques, cuscuz, jian, lokum, pacoca (Paçoca), patua (Patuá), saci.

Each image folder also contains a `generation_manifest.csv` with the seed and prompt used per image.

## Content note on the Textual Inversion baseline

For several concepts, the Textual Inversion baseline's learned embedding did not converge to the
target concept and instead drifted toward generating people, in a fraction of cases including nudity
(this is reported and discussed as a finding in the paper's Textual Inversion comparison section).
Images in `textual_inversion_<concept>_with_baseline/` were screened with the standard Stable
Diffusion safety checker (`CompVis/stable-diffusion-safety-checker`) and flagged images were removed
before upload (excluded counts, out of 1000 generated per concept: chamanto 40, chaneques 196,
cuscuz 198, jian 13, lokum 141, pacoca 454, patua 489, saci 115). This automated filter is not perfect
(a manual spot-check found a small number of missed borderline images among the remaining ones for
`pacoca` and `patua`, which is expected given the high underlying rate for those two concepts) and this
data should not be treated as fully verified for all downstream uses; it is intended to support the
quantitative comparison and the observation that Textual Inversion, in this configuration, did not
reliably learn several of the target concepts. The quantitative metrics for the Textual Inversion
baseline reported in the paper (Table 3: CLIP Score, FID, KID) are computed on these
safety-filtered images (i.e., the images present in this dataset); the DreamBooth+LoRA metrics are
computed on the full 1000-image sets, since those required no filtering.
