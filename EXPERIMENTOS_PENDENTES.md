# Experimentos pendentes — revisão MTAP (MTAP-D-25-02267_R1)

Este documento lista tudo o que ainda precisa rodar no servidor para que o paper
(`paper_tex/sn-article.tex`) cumpra o que a carta de resposta (`paper_tex/rev2.tex`) promete.

| Pedido na carta | Revisor | Produzido por |
|---|---|---|
| Baseline com Textual Inversion (SDXL), mesmos dados, números + exemplos visuais | R2-C1 | `baseline.py` + `compare_methods.py` |
| 1000 imagens por conceito (em vez de 100) | R2-C3 | `experiments.py` / `baseline.py` (`--num-images 1000`) |
| KID além de CLIP e FID | R2-C3 | `experiment_utils.py` |
| Média, desvio e IC95% por bootstrap | R2-C3, R3-C2 | `experiment_utils.py` |
| Análise por grupo (comida, folclore, artefatos) | R2-C5, R7-C2 | `aggregate_metrics_by_group` |
| Refazer a Fig. 3 com todos os conceitos e números consistentes com a Tabela 2 | pendência interna | `experiments.py` (checkpoints) |

**Ordem:** Parte 0 (corrigir código) → Parte 1 (servidor) → Parte 2 (rodar) → Parte 3 (trazer resultados) → Parte 4 (atualizar paper e carta).

---

## Parte 0 — Correções obrigatórias no código ANTES de subir

Os scripts atuais **não rodam como estão**, ou rodam de forma inviável. Tudo abaixo foi verificado lendo o código
e o script `examples/textual_inversion/textual_inversion_sdxl.py` do diffusers v0.30.3 (a versão fixada em `requirements.txt`).

### 0.1 `baseline.py` quebra logo no início do treino (argumentos inexistentes)
O script de Textual Inversion SDXL do diffusers v0.30.3 **não aceita**:
- `--pretrained_vae_model_name_or_path` (passado em `baseline.py:145`)
- `--validation_epochs` (passado em `baseline.py:160`). Ele só aceita `--validation_steps`.

Como o argparse recusa argumentos desconhecidos, o treino morre na hora.
**Correção:** remover o argumento do VAE e trocar `--validation_epochs=25` por `--validation_steps=500` (ou remover a validação).

### 0.2 Textual Inversion em fp16 produz NaN (VAE do SDXL)
Sem o argumento de VAE, o script usa o VAE original do SDXL e o converte para `weight_dtype`. O VAE original do SDXL
**gera NaN em fp16**, e é por isso que o DreamBooth usa `madebyollin/sdxl-vae-fp16-fix`.
**Correção:** rodar o baseline com `--mixed-precision bf16` (a A40 suporta bf16).

### 0.3 `--kid-subset-size 50` quebra o KID
O KID do torchmetrics exige `subset_size` ≤ número de amostras. O conjunto real tem **20 imagens (19 no Saci)**.
**Correção:** usar `--kid-subset-size 19` nos dois scripts (ou mudar o default).

### 0.4 FID/KID com bootstrap é inviável do jeito que está (CPU + recarrega a Inception)
Em `experiment_utils.py`, `bootstrap_distribution` chama `compute_fid` e `compute_kid` **200 vezes cada**.
Cada chamada instancia uma `FrechetInceptionDistance` ou `KernelInceptionDistance` nova (recarrega a InceptionV3),
**na CPU**, e reprocessa ~1020 imagens. São ~400 mil passadas da InceptionV3 por pasta, em ~130 pastas: semanas de processamento.
**Correção:**
1. Extrair as features da InceptionV3 (2048-d) **uma única vez** por pasta, **na GPU**. As features reais podem ficar em cache por conceito.
2. Fazer o bootstrap reamostrando **as features**: FID = fórmula de Fréchet com `scipy.linalg.sqrtm`, KID = MMD com kernel polinomial cúbico.

### 0.5 CLIP Score recarrega o modelo CLIP a cada imagem
`compute_clip_statistics` chama `torchmetrics.functional.multimodal.clip_score(..., model_name_or_path=...)` por imagem,
o que recarrega o CLIP ViT-B/16 a cada chamada, na CPU.
**Correção:** instanciar `CLIPScore(model_name_or_path="openai/clip-vit-base-patch16").to("cuda")` uma vez e processar em lotes.
Manter o **mesmo modelo** (ViT-B/16) e o **mesmo texto** (descrições da Tabela 1), que é o que o paper descreve.

### 0.6 `compare_methods.py` nunca compara DreamBooth com Textual Inversion
O join é feito por `(concept, condition)`. A condição do DreamBooth é `with_finetuning`, e a do baseline é `with_baseline`.
A única condição em comum é `without_finetuning`, então a tabela de comparação sai **só com o modelo base contra ele mesmo**.
**Correção:** juntar `with_finetuning` (DreamBooth) com `with_baseline` (TI), por exemplo mapeando as duas para uma condição comum `fine_tuned`.

### 0.7 IC95% por grupo não faz sentido com 2–3 conceitos
`aggregate_metrics_by_group` faz bootstrap sobre as **médias dos conceitos**. O grupo folclore tem só 2 conceitos (Saci, Chaneques),
então o "IC" sai degenerado.
**Decisão (escolher uma):**
- (a) reportar por grupo a média e o intervalo mín–máx dos conceitos, sem IC; **ou**
- (b) bootstrap juntando as imagens de todos os conceitos do grupo (CLIP) e as features (FID/KID).

### 0.8 Trabalho duplicado (opcional, economiza horas e disco)
- Em `experiments.py`, `with_finetuning` usa o checkpoint 3000 **com as mesmas seeds** de `checkpoint_3000`, e gera 8.000 imagens idênticas duas vezes.
- `baseline.py` gera de novo as imagens do modelo base (`without_finetuning`) com **outras seeds** (1000 contra 2000). São 8.000 imagens a mais, e o "modelo base" fica com números diferentes em cada CSV.
**Correção sugerida:** no baseline, apontar para as pastas `without_finetuning` do DreamBooth em vez de gerar de novo.
No DreamBooth, criar `with_finetuning` como symlink de `checkpoint_3000`.

---

## Parte 1 — Preparar o servidor

### 1.1 Hardware
- Os treinos originais rodaram em **1× NVIDIA A40 (48 GB)**, 96 CPUs lógicas e ~540 GB de RAM (dados do wandb).
- O DreamBooth-LoRA usou pico de **~27 GB de VRAM**. Uma GPU de 24 GB provavelmente não aguenta essa configuração.
- **Disco:** cada PNG 1024×1024 tem ~1,5 MB. Com 1000 imagens por pasta:
  - DreamBooth: 8 conceitos × 7 pastas (base + 6 checkpoints) ≈ 56 mil imagens ≈ **85 GB**
  - Textual Inversion final: 8 × 1000 ≈ **12 GB** (+ ~70 GB se gerar também os checkpoints do TI)
  - **Reserve ~100–200 GB.**

> Se o servidor for diferente da A40, anote GPU, CPU, RAM, tempo e pico de VRAM. O parágrafo *Hardware* do paper precisa refletir a máquina da nova avaliação.

### 1.2 Ambiente
```bash
cd /workspace
git clone https://github.com/Fmahlow/Auto-Tune.git
git clone --branch v0.30.3 --depth 1 https://github.com/huggingface/diffusers.git
export DIFFUSERS_REPO=/workspace/diffusers

python -m venv /workspace/venv && source /workspace/venv/bin/activate
# Fixar torch/xformers compatíveis (requirements.txt não fixa torch, o que já causou erro de import do diffusers)
pip install torch==2.4.1 torchvision==0.19.1 xformers==0.0.28.post1 --index-url https://download.pytorch.org/whl/cu124
pip install -r Auto-Tune/requirements.txt
pip install wandb && wandb login          # para registrar tempo e memória do TI também
accelerate config default                  # 1 GPU, sem DeepSpeed
```
Teste rápido do ambiente:
```bash
python -c "from diffusers import DiffusionPipeline, LCMScheduler; import torchmetrics, peft, bitsandbytes; print('ok')"
```

### 1.3 Pasta de dados limpa
Os scripts descobrem conceitos varrendo `--data-root`. Use uma pasta só com os 8 conceitos:
```bash
mkdir -p /workspace/data
for c in chamanto chaneques cuscuz jian lokum paçoca patuá saci; do
  ln -s "/workspace/Auto-Tune/$c" "/workspace/data/$c"
done
find -L /workspace/data -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' -o -iname '*.webp' \) | wc -l   # esperado: 159 (7×20 + 19 do Saci)
```

---

## Parte 2 — Rodar

Rode tudo dentro de `tmux` ou `nohup`. Acompanhe com:
```bash
python /workspace/Auto-Tune/watch_progress.py /workspace/outputs/dreambooth/progress.json
```

**Recomendação:** separar geração e avaliação (`--skip-eval` primeiro, depois `--skip-train --skip-generate`).
Assim um erro na avaliação não obriga a gerar tudo de novo.

### Decisão A — Reaproveitar os LoRAs antigos ou treinar de novo?
Se o servidor antigo ainda tiver `/workspace/testing_fine_tuning/`, os checkpoints originais podem ser reaproveitados.
Isso economiza ~12 h e mantém o mesmo modelo das Figs. 1 e 2. Mapeamento (tirado do `experiments.ipynb`):

| Conceito (`safe_name`) | Pasta antiga |
|---|---|
| patua | `17407661407321577` |
| cuscuz | `17407600811423647` |
| chaneques | `17407540905371664` |
| chamanto | `1740748525261208` |
| lokum | `174074268215229` |
| pacoca | `17406920505560935` |
| jian | `17406842774634914` |
| saci | `17406783660950785` |

```bash
OLD=/workspace/testing_fine_tuning
NEW=/workspace/outputs/dreambooth/training_runs
mkdir -p $NEW
declare -A M=([patua]=17407661407321577 [cuscuz]=17407600811423647 [chaneques]=17407540905371664 \
  [chamanto]=1740748525261208 [lokum]=174074268215229 [pacoca]=17406920505560935 \
  [jian]=17406842774634914 [saci]=17406783660950785)
for k in "${!M[@]}"; do ln -s "$OLD/${M[$k]}" "$NEW/$k"; done
ls $NEW/*/checkpoint-3000/pytorch_lora_weights.safetensors   # devem existir 8
```
Se reaproveitar, rode com `--skip-train`. Se não existirem, retreine (remova `--skip-train`, ~85–91 min por conceito na A40).

### 2.1 DreamBooth + LoRA — geração (1000 imagens: base + 6 checkpoints)
```bash
cd /workspace/Auto-Tune
python experiments.py \
  --data-root /workspace/data \
  --output-root /workspace/outputs/dreambooth \
  --enable-xformers --use-8bit-adam \
  --num-images 1000 \
  --kid-subset-size 19 \
  --extra-train-arg=--report_to=wandb \
  --skip-train \
  --skip-eval
```
(Sem `--skip-train` se for retreinar.)

### 2.2 Textual Inversion (baseline) — treino + geração
Mesmos dados, 3000 passos, batch 1, acumulação 2, resolução 1024 e mesma inferência (LCM, 4 passos, guidance 8).
```bash
python baseline.py \
  --data-root /workspace/data \
  --output-root /workspace/outputs/textual_inversion \
  --enable-xformers \
  --mixed-precision bf16 \
  --num-images 1000 \
  --kid-subset-size 19 \
  --checkpoints \
  --extra-train-arg=--report_to=wandb \
  --skip-eval
```
- `--checkpoints` sem valores = não gera imagens por checkpoint do TI (a carta só pede a comparação final). Remova se quiser a curva do TI também.
- `--initializer-token` está como `object` para todos. Opcional: usar um token por grupo (`food`, `creature`, `object`) é uma inicialização mais justa.
- Anote tempo e pico de VRAM do TI (ficam no wandb) para o paper.

### 2.3 Avaliação (CLIP, FID, KID com IC95%)
```bash
python experiments.py --data-root /workspace/data --output-root /workspace/outputs/dreambooth \
  --skip-train --skip-generate --kid-subset-size 19 --bootstrap-samples 1000
python baseline.py --data-root /workspace/data --output-root /workspace/outputs/textual_inversion \
  --skip-train --skip-generate --kid-subset-size 19 --bootstrap-samples 1000 --checkpoints
```
Com a correção 0.4, `--bootstrap-samples 1000` fica barato (só álgebra sobre features). Sem ela, **não rode**.

### 2.4 Comparação entre métodos
```bash
python compare_methods.py \
  --dreambooth-root /workspace/outputs/dreambooth \
  --baseline-root /workspace/outputs/textual_inversion \
  --output-root /workspace/outputs/comparison
```
Antes, confira se a correção 0.6 foi aplicada: `comparison_individual.csv` precisa ter linhas de **fine-tuned**, não só `without_finetuning`.

### Estimativa de tempo (A40, **estimativa**, confirme medindo as primeiras 100 imagens)
| Etapa | Volume | Tempo aproximado |
|---|---|---|
| Geração SDXL + LCM (4 passos, 1024²) | ~0,6–1,0 s/imagem | — |
| DreamBooth: base + 6 checkpoints | 56 mil imagens | ~10–16 h |
| TI: treino | 8 × 3000 passos | ~8–12 h (medir) |
| TI: geração final | 8 mil imagens | ~1,5–2,5 h |
| Retreino DreamBooth (se Decisão A = retreinar) | 8 × 3000 passos | ~12 h |
| Avaliação (após 0.4/0.5) | ~64 pastas | poucas horas |

---

## Parte 3 — O que trazer de volta do servidor

Não precisa baixar as ~100 GB de imagens. Traga:
- `outputs/dreambooth/metrics_dreambooth.csv`, `metrics_checkpoints_dreambooth.csv`, `metrics_groupwise_*.csv`
- `outputs/textual_inversion/metrics_baseline.csv`, `metrics_groupwise_baseline.csv`
- `outputs/comparison/comparison_individual.{csv,md}`, `comparison_groupwise.{csv,md}`
- `outputs/*/qualitative/*.png` (folhas com exemplos visuais para a figura DreamBooth × TI)
- `outputs/*/progress.log`
- `outputs/*/human_eval/` (pacote cego para avaliação humana futura; não é usado agora)
- Algumas imagens por condição para montar a figura qualitativa (por exemplo, as das folhas acima)
- Tempo e pico de VRAM dos treinos de TI (wandb)

---

## Parte 4 — O que atualizar no paper e na carta depois de rodar

**Paper (`sn-article.tex`)**
1. **Tabela 2:** trocar pelos novos valores (1000 imagens), com CLIP, FID e KID em média ± desvio e IC95%. Atualizar a legenda ("100 generated images" → 1000).
2. **Fig. 3:** refazer com os novos CSVs de checkpoint. Isso resolve as pendências: (i) **curva do Patuá ausente na Fig. 3(b)**; (ii) **FID do checkpoint 3000 diferente da Tabela 2** (agora vem das mesmas imagens).
3. **Parágrafos da análise por checkpoint:** o texto atual cita números da execução antiga (ex.: Paçoca 28,81 → 24,42; melhor CLIP em 2000 passos para 4 conceitos). **Reescrever com os números novos.**
4. **Nova subseção:** comparação DreamBooth+LoRA × Textual Inversion (tabela + figura qualitativa), com a configuração do TI em Metodologia.
5. **Nova tabela ou parágrafo:** análise por grupo (comida / folclore / artefatos).
6. **Seção Métricas:** adicionar KID (definição + referência Bińkowski et al., 2018), bootstrap e número de amostras. Mencionar que a referência de FID/KID são as 19–20 imagens de treino, e que por isso o FID absoluto é pouco confiável (com 20 amostras reais a covariância 2048-d é singular). O KID é mais adequado nesse regime.
7. **Hardware:** acrescentar a máquina e o tempo da nova avaliação e do TI.
8. **Limitações:** hoje o texto não fala de testes estatísticos. Com IC95%, dá para dizer que as diferenças são (ou não) consistentes.

**Carta (`rev2.tex`)**
- Trocar "10 images" por 20 (19 no Saci) nas respostas R2-C2, R3-C2 e R8-C1.
- R3-C2 diz que teste estatístico é trabalho futuro, mas R2-C3 diz que os IC95% foram adicionados. Alinhar as duas.
- Indicar seção/tabela/figura de cada mudança.
- R2-C4: citar A40, pico de VRAM e tempo por treino.

**Ainda aberto, independe de rodar**
- Decidir o esquema de cores final (azul = mudanças desta rodada; vermelho = texto anterior) conforme a MTAP.
- Compilar no Overleaf.
