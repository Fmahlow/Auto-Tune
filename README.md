# Automated Fine-Tuning of Diffusion Models for Learning Underrepresented Concepts from Web-Scraped Data

This repository contains the scripts, data, and experimental outputs associated with the paper **"Automated Fine-Tuning of Diffusion Models for Learning Underrepresented Concepts from Web-Scraped Data"** (Submitted to IEEE Computer Graphics and Applications). In this work, we present an automated pipeline to fine-tune diffusion-based models on culturally specific visual concepts that are typically underrepresented in mainstream datasets.

> ⚠️ **Disclaimer:** This repository is **not organized with reproducibility as its main goal**. It serves as a **record of experiments** conducted during the project. We plan to refactor and release a cleaner, fully documented version of the codebase in the future.

---

## 🧪 Project Overview

Our goal was to train generative models (e.g., **Stable Diffusion XL**) to learn niche visual concepts—such as folklore characters, traditional foods, and cultural artifacts—by collecting training images directly from the web. The pipeline includes:

- Automated image scraping using Selenium
- Model fine-tuning with LoRA
- Image generation accelerated by Latent Consistency Models (LCMs)
- Evaluation using CLIP Score and Fréchet Inception Distance (FID)

---

## 📁 Repository Structure

- **Folders like `saci`, `patuá`, `lokum`, etc.**  
  These contain **training images collected via web scraping**, used for fine-tuning each specific concept.

- **CSV files**  
  Files such as `clip_score.csv`, `fid_scores_results.csv`, and their `_checkpoints` versions contain **quantitative metrics** computed during training, including **CLIP Scores** (semantic alignment) and **FID scores** (visual similarity to real data).

- **Figures (`.pdf`, `.png`)**  
  Files like `fig_1_paper.pdf`, `fig_2_paper.pdf`, and `figura_resultante.png` include **visualizations used in the paper**, showcasing generated images before and after fine-tuning, as well as performance trends over training steps.
---

## 📓 Main File: `experiments.ipynb`

The core of the project is the notebook `experiments.ipynb`. It contains:

1. Automated image collection for culturally specific prompts using Selenium
2. Fine-tuning of Stable Diffusion XL using LoRA and custom datasets
3. Image generation before and after training for visual comparison
4. Quantitative evaluation using CLIP Score and FID at multiple training checkpoints (500–3000 steps)
5. Plotting and saving of the figures used in the final publication

---

## ✍️ Authors

- Felipe Mahlow  
- Renato Dias de Souza  
- João Paulo Papa  
- Kelton Augusto Pontara da Costa

---

## 📌 Future Work

We aim to:
- [ ] Refactor the code into a reproducible and modular pipeline  
- [ ] Add documentation and usage instructions  
- [ ] Enable easier integration of new concepts and data sources  
- [ ] Expand the system to include bias diagnostics and cultural sensitivity validation

---

Feel free to explore, experiment, and reach out with suggestions or feedback.
